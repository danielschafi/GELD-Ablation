from logging import getLogger

import torch

from TSP_Model import TSPModel as Model
from TSPEnv import TSPEnv as Env
from utils.beam_search import Beamsearch
from utils.utils import *


class TSPTester:
    def __init__(self, env_params, model_params, tester_params, ablation_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.ablation_params = ablation_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.dtypeFloat = torch.cuda.FloatTensor
        self.dtypeLong = torch.cuda.LongTensor

        # cuda
        USE_CUDA = self.tester_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params["model_load"]
        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self, size=None, disribution=None):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        self.env.load_raw_data(
            self.tester_params["test_episodes"],
            SL_Test=True,
            MSV=True,
            size=size,
            disribution=disribution,
        )

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        test_num_episode = self.tester_params["test_episodes"]
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params["test_batch_size"], remaining)

            score_teacher, score_student, problems_size = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2
            )

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode, test_num_episode
            )
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                    episode,
                    test_num_episode,
                    elapsed_time_str,
                    remain_time_str,
                    score_teacher,
                    score_student,
                )
            )

            all_done = episode == test_num_episode

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(
                    " Gap: {:.4f}%".format(
                        (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                    )
                )
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(
        self,
        after_repair_sub_solution,
        before_reward,
        after_reward,
        indices,
        len_of_sub,
        solution,
    ):
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        origin_sub_solution = solution[:, indices]
        jjj, _ = torch.sort(origin_sub_solution, dim=-1, descending=False)
        kkk_2 = jjj.gather(2, after_repair_sub_solution.view(jjj.shape))
        if_repair = before_reward > after_reward
        if_repair = if_repair.unsqueeze(1).view(jjj.shape[0], jjj.shape[1])
        temp_result = solution[:, indices].clone()
        temp_result[if_repair] = kkk_2[if_repair]
        solution[:, indices] = temp_result
        return solution

    def _solve_batch_beam_search(self, batch_size, problem_size):
        """Stage 1: Beam Search.

        Solves the TSP Problem using beam search based on distance matrix instead of probabilities outputted by the model.
        """

        # Setup beam search
        beam_size = self.tester_params["beam_size"]
        beamsearch = Beamsearch(
            beam_size,
            batch_size,
            problem_size,
            self.dtypeFloat,
            self.dtypeLong,
            probs_type="logits",
            random_start=False,
            device=self.origin_problem.device,
        )

        current_step = 0
        self.env.reset(batch_size * beam_size)
        # state is the problems [B*beam_size, problem_size, 2]
        state, reward, reward_student, done = self.env.pre_step()

        # Normalize
        data = state.data
        min_val, _ = torch.min(data, dim=1, keepdim=True)
        max_val, _ = torch.max(data, dim=1, keepdim=True)
        max_diff_values, _ = torch.max(max_val - min_val, dim=-1)
        norm_data = (data - min_val) / max_diff_values.unsqueeze(2)
        self.data = norm_data

        # Compute distance matrix: (batch_size, N, N) — same for all beams
        self.dis_matrix = torch.cdist(self.data, self.data, p=2)
        # Fill diagonal with a large value — NOT zero. If diagonal = 0 then
        # trans_probs[self] = -0 = 0, and the logits mask (×1e20) gives 0×1e20 = 0,
        # which is the highest value and topk would re-select the current node every step.
        self.dis_matrix.diagonal(dim1=-2, dim2=-1).fill_(1e9)

        while not done:
            if current_step == 0:
                selected_teacher = torch.zeros(
                    batch_size * beam_size, dtype=torch.int64
                )
            else:
                # Slice distances from the current node for each beam hypothesis:
                # current_nodes: (batch_size, beam_size)
                # dis_4d:        (batch_size, beam_size, N, N)
                # trans_probs:   (batch_size, beam_size, N)  — negated so topk picks shortest
                current_nodes = beamsearch.next_nodes[-1]
                # Expand (B, N, N) → (B, beam, N, N); all beams share the same distances
                dis_4d = self.dis_matrix.unsqueeze(1).expand(
                    batch_size, beam_size, problem_size, problem_size
                )
                # Build index (B, beam, 1, N): repeat the current-node index across all N destinations,
                # then gather picks the single row dis[b, k, current_node, :] for each hypothesis
                idx = (
                    current_nodes.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(batch_size, beam_size, 1, problem_size)
                )
                trans_probs = -dis_4d.gather(2, idx).squeeze(
                    2
                )  # negate: topk(largest) → shortest edge
                self.env.selected_node_list = beamsearch.advance(
                    trans_probs, self.env.selected_node_list
                )
                selected_teacher = beamsearch.next_nodes[-1].view(-1)

            state, reward, done = self.env.step_beam(selected_teacher, beam=beam_size)
            current_step += 1

        reward = reward.view(batch_size, beam_size)
        selected_node_list = self.env.selected_node_list.view(batch_size, beam_size, -1)
        current_best_length, min_idx = reward.min(1)
        zero_to_bsz = torch.arange(
            batch_size, dtype=torch.long, device=current_best_length.device
        )
        best_select_node_list = selected_node_list[zero_to_bsz, min_idx]
        return best_select_node_list, current_best_length

    def _solve_batch_knn_beam_search(self, batch_size, problem_size):
        """Stage 1: k-NN constrained Beam Search.

        Like _solve_batch_beam_search but each step only considers the k
        geometrically nearest nodes as candidates; all others are blocked.
        """
        beam_size = self.tester_params["beam_size"]
        k = self.tester_params.get("knn_k", 99)

        beamsearch = Beamsearch(
            beam_size,
            batch_size,
            problem_size,
            self.dtypeFloat,
            self.dtypeLong,
            probs_type="logits",
            random_start=False,
            device=self.origin_problem.device,
        )

        current_step = 0
        self.env.reset(batch_size * beam_size)
        state, reward, reward_student, done = self.env.pre_step()

        # Normalize
        data = state.data
        min_val, _ = torch.min(data, dim=1, keepdim=True)
        max_val, _ = torch.max(data, dim=1, keepdim=True)
        max_diff_values, _ = torch.max(max_val - min_val, dim=-1)
        norm_data = (data - min_val) / max_diff_values.unsqueeze(2)
        self.data = norm_data

        # Compute distance matrix: (batch_size, N, N) — same for all beams
        self.dis_matrix = torch.cdist(self.data, self.data, p=2)
        # Same diagonal fix as in _solve_batch_beam_search — see comment there.
        self.dis_matrix.diagonal(dim1=-2, dim2=-1).fill_(1e9)

        # Pre-compute k-NN lookup once — fill diagonal with inf so a node is
        # never its own nearest neighbour, then take the k smallest per row.
        # knn_4d: (B, beam, N, k)
        dis_for_knn = self.dis_matrix.clone()
        dis_for_knn.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        _, knn_indices = torch.topk(dis_for_knn, k, dim=2, largest=False)
        knn_4d = knn_indices.unsqueeze(1).expand(batch_size, beam_size, problem_size, k)

        while not done:
            if current_step == 0:
                selected_teacher = torch.zeros(
                    batch_size * beam_size, dtype=torch.int64
                )
            else:
                current_nodes = beamsearch.next_nodes[-1]  # (B, beam)

                # Full negative-distance row for the current node — same gather
                # pattern as _solve_batch_beam_search
                dis_4d = self.dis_matrix.unsqueeze(1).expand(
                    batch_size, beam_size, problem_size, problem_size
                )
                idx_row = (
                    current_nodes.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(batch_size, beam_size, 1, problem_size)
                )
                neg_dist_full = -dis_4d.gather(2, idx_row).squeeze(2)  # (B, beam, N)

                # Look up the k nearest neighbours of the current node, then
                # scatter their distances into an otherwise -1e20 tensor so
                # the beam can only advance to those k candidates.
                idx_knn = (
                    current_nodes.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(batch_size, beam_size, 1, k)
                )
                knn_for_current = knn_4d.gather(2, idx_knn).squeeze(2)  # (B, beam, k)
                trans_probs = torch.full_like(neg_dist_full, -1e20)
                trans_probs.scatter_(
                    2, knn_for_current, neg_dist_full.gather(2, knn_for_current)
                )

                self.env.selected_node_list = beamsearch.advance(
                    trans_probs, self.env.selected_node_list
                )
                selected_teacher = beamsearch.next_nodes[-1].view(-1)

            state, reward, done = self.env.step_beam(selected_teacher, beam=beam_size)
            current_step += 1

        reward = reward.view(batch_size, beam_size)
        selected_node_list = self.env.selected_node_list.view(batch_size, beam_size, -1)
        current_best_length, min_idx = reward.min(1)
        zero_to_bsz = torch.arange(
            batch_size, dtype=torch.long, device=current_best_length.device
        )
        best_select_node_list = selected_node_list[zero_to_bsz, min_idx]
        return best_select_node_list, current_best_length

    def _test_one_batch(self, episode, batch_size, clock=None):

        # ablation_params = {
        #     "stage_1": "BeamSearch", # One of BeamSearch, knn-BeamSearch, Neural
        #     "stage_2": None # One of None, BeamSearch-RC, Neural-RC
        # }

        assert self.ablation_params["stage_2"] in (
            None,
            "Neural-RC",
            "BeamSearch-RC",
        ), f"unknown stage_2: {self.ablation_params['stage_2']}"

        self.model.eval()
        with torch.no_grad():
            # Load a batch of problems and solutions into self.env.problems, self.env.solution
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            self.optimal_length = self.env._get_travel_distance_2(
                self.origin_problem, self.env.solution
            )

            name = "TSP" + str(self.origin_problem.shape[1])
            B_V = batch_size * 1
            problem_size = self.origin_problem.shape[1]

            ##################
            ####  Stage 1 ####
            ##################

            if self.ablation_params["stage_1"] == "BeamSearch":
                best_select_node_list, current_best_length = (
                    self._solve_batch_beam_search(batch_size, problem_size)
                )

            if self.ablation_params["stage_1"] == "knn-BeamSearch":
                best_select_node_list, current_best_length = (
                    self._solve_batch_knn_beam_search(batch_size, problem_size)
                )

            elif self.ablation_params["stage_1"] == "Neural":
                # Solving it using GELD model

                # greedy

                # prepare env
                current_step = 0
                reset_state, _, _ = self.env.reset()
                state, reward, reward_student, done = (
                    self.env.pre_step()
                )  # state: data, first_node = current_node
                self.model.pre_forward(state=state)

                while not done:
                    if current_step == 0:
                        selected_teacher = torch.zeros(B_V, dtype=torch.int64)
                        selected_student = selected_teacher
                    else:
                        # Model predict next node
                        selected_teacher, _, _, selected_student = self.model(
                            state,
                            self.env.selected_node_list,
                            self.env.solution,
                            current_step,
                        )
                    current_step += 1
                    # Greedy Step

                    # make step in environment, get reward,
                    # -> append node to list.
                    # when done then compute reward and return
                    state, reward, reward_student, done = self.env.step(
                        selected_teacher, selected_student
                    )

                best_select_node_list = self.env.selected_node_list
                current_best_length = self.env._get_travel_distance_2(
                    self.origin_problem, best_select_node_list
                )
                torch.cuda.empty_cache()

                # beam
                if self.tester_params["beam"]:
                    beam_size = self.tester_params["beam_size"]
                    beamsearch = Beamsearch(
                        beam_size,
                        batch_size,
                        problem_size,
                        self.dtypeFloat,
                        self.dtypeLong,
                        probs_type="logits",
                        random_start=False,
                        device=self.origin_problem.device,
                    )
                    current_step = 0
                    self.env.reset(batch_size * beam_size)
                    state, reward, reward_student, done = self.env.pre_step()
                    while not done:
                        if current_step == 0:
                            selected_teacher = torch.zeros(
                                batch_size * beam_size, dtype=torch.int64
                            )
                        else:
                            _, trans_probs, _, _ = self.model(
                                state,
                                self.env.selected_node_list,
                                self.env.solution,
                                current_step,
                                beam_search=True,
                                beam_size=beam_size,
                            )
                            probs = torch.log(
                                trans_probs.view(batch_size, beam_size, -1)
                            )
                            probs[probs.isnan()] = 0
                            self.env.selected_node_list = beamsearch.advance(
                                probs, self.env.selected_node_list
                            )
                            selected_teacher = beamsearch.next_nodes[-1].view(-1)

                            # Beam Search Step
                        state, reward, done = self.env.step_beam(
                            selected_teacher, beam=beam_size
                        )
                        current_step += 1
                    reward = reward.view(batch_size, beam_size)
                    selected_node_list = self.env.selected_node_list.view(
                        batch_size, beam_size, -1
                    )
                    current_best_length, min_idx = reward.min(1)
                    zero_to_bsz = torch.arange(
                        batch_size, dtype=torch.long, device=current_best_length.device
                    )
                    best_select_node_list = selected_node_list[zero_to_bsz, min_idx]
                torch.cuda.empty_cache()

            ##################
            ####  Stage 2 ####
            ##################
            if self.ablation_params["stage_2"] is not None:
                # PRC
                if self.tester_params["PRC"]:
                    origin_problem = self.env.problems[:]
                    num_RC = self.tester_params["num_PRC"]
                    sample_max = problem_size // 4
                    for step_RC in range(num_RC):
                        val_num_samples = torch.randint(
                            low=2, high=sample_max + 1, size=[1]
                        )[0]
                        max_lenth = problem_size // val_num_samples
                        interval = problem_size // val_num_samples
                        if step_RC % 2 != 0:
                            best_select_node_list = torch.flip(
                                best_select_node_list, dims=[1]
                            )
                        best_select_node_list = best_select_node_list.roll(
                            dims=1,
                            shifts=int(
                                torch.randint(low=0, high=problem_size, size=[1])[0]
                            ),
                        )
                        if_rotation = torch.randint(low=0, high=8, size=[1])[0]
                        if if_rotation != 0:
                            x = origin_problem[:, :, [0]]
                            y = origin_problem[:, :, [1]]
                            if if_rotation == 1:
                                origin_problem = torch.cat((1 - x, y), dim=2)
                            elif if_rotation == 2:
                                origin_problem = torch.cat((x, 1 - y), dim=2)
                            elif if_rotation == 3:
                                origin_problem = torch.cat((1 - x, 1 - y), dim=2)
                            elif if_rotation == 4:
                                origin_problem = torch.cat((y, x), dim=2)
                            elif if_rotation == 5:
                                origin_problem = torch.cat((1 - y, x), dim=2)
                            elif if_rotation == 6:
                                origin_problem = torch.cat((y, 1 - x), dim=2)
                            elif if_rotation == 7:
                                origin_problem = torch.cat((1 - y, 1 - x), dim=2)

                        select_node_list = best_select_node_list[:]
                        indices = torch.arange(
                            0, problem_size, step=interval, dtype=torch.long
                        )[:val_num_samples]
                        len_of_sub = torch.randint(low=4, high=max_lenth + 1, size=[1])[
                            0
                        ]
                        new_problem, new_solution = self.sampling_subpaths_p(
                            origin_problem, select_node_list, indices, len_of_sub
                        )
                        # print("Original rig")
                        # print(origin_problem)
                        self.env.problems = new_problem.view(-1, len_of_sub, 2)
                        self.env.solution = new_solution.view(-1, len_of_sub)
                        self.env.batch_size = self.env.problems.size(0)
                        partial_solution_length = self.env._get_travel_distance_2(
                            self.env.problems, self.env.solution
                        )
                        if self.ablation_params["stage_2"] == "Neural-RC":
                            # Greedy re-combination (original PRC path)
                            self.env.reset()
                            state, _, _, done = self.env.pre_step()
                            self.model.pre_forward(state=state)
                            current_step = 0
                            while not done:
                                if current_step == 0:
                                    selected_teacher = self.env.solution[:, -1]
                                    selected_student = self.env.solution[:, -1]
                                elif current_step == 1:
                                    selected_teacher = self.env.solution[:, 0]
                                    selected_student = self.env.solution[:, 0]
                                else:
                                    selected_teacher, _, _, selected_student = (
                                        self.model(
                                            state,
                                            self.env.selected_node_list,
                                            self.env.solution,
                                            current_step,
                                            repair=True,
                                        )
                                    )

                                current_step += 1
                                state, reward, reward_student, done = self.env.step(
                                    selected_teacher, selected_student
                                )
                            ahter_repair_sub_solution = torch.roll(
                                self.env.selected_node_list, shifts=-1, dims=1
                            )
                            after_repair_reward = reward_student

                        else:
                            # BeamSearch-RC: distance-driven beam search over the sub-problem.
                            # No model is consulted; scores are negated pairwise distances.
                            # PRC pins the two open-path endpoints: the new sub-tour must
                            # start at `head` (= solution[:, 0]) and end at `tail` (= solution[:, -1])
                            # so it can replace the original sub-path in the full tour.
                            #
                            # Implementation: pre-mask both endpoints, start the beam at tail
                            # (via start_nodes), and force every beam to select head on the very
                            # first advance call by feeding a one-hot trans_probs. From step 2
                            # onward, scores are -distance(current_node, *).
                            beam_size = self.tester_params["beam_size"]
                            batch_sub = self.env.problems.size(0)
                            len_sub = int(len_of_sub)
                            device = self.origin_problem.device

                            tail = self.env.solution[:, -1]  # (batch_sub,)
                            head = self.env.solution[:, 0]  # (batch_sub,)

                            # Expand env to (batch_sub * beam_size) for beam decoding
                            self.env.reset(batch_sub * beam_size)
                            state, _, _, done = self.env.pre_step()

                            # Distance matrix on the *original* sub-problem coords
                            # (no normalization — keeps lengths directly comparable with
                            # partial_solution_length, which is also on un-normalized coords).
                            data = self.env.problems  # (batch_sub, len_sub, 2)
                            dis_matrix = torch.cdist(data, data, p=2)
                            # Prevent self-loops: a 0 entry would be multiplied by 1e20 in the
                            # logits mask -> 0, which would tie for top-k. See
                            # _solve_batch_beam_search for the same fix.
                            dis_matrix.diagonal(dim1=-2, dim2=-1).fill_(1e9)

                            beamsearch = Beamsearch(
                                beam_size,
                                batch_sub,
                                len_sub,
                                self.dtypeFloat,
                                self.dtypeLong,
                                probs_type="logits",
                                random_start=False,
                                device=device,
                            )

                            # Overwrite the default start_nodes (zeros) with `tail`. We must
                            # also rebuild the mask from scratch — the constructor masked
                            # node 0; we need only `tail` masked at this point.
                            tail_be = (
                                tail.view(-1, 1).expand(-1, beam_size).contiguous()
                            )
                            head_be = (
                                head.view(-1, 1).expand(-1, beam_size).contiguous()
                            )
                            beamsearch.start_nodes = tail_be.clone()
                            beamsearch.next_nodes = [tail_be.clone()]
                            beamsearch.mask = torch.ones(
                                batch_sub, beam_size, len_sub, device=device
                            ).type(self.dtypeFloat)
                            beamsearch.update_mask(tail_be)

                            # env now needs the tail column in its history so step_beam's
                            # done check (selected_count == problem_size) lines up after
                            # `len_sub - 1` advance/step_beam pairs below.
                            self.env.selected_node_list = tail.repeat_interleave(
                                beam_size
                            ).unsqueeze(1)
                            self.env.selected_count = 1

                            current_step = 1
                            done = False
                            while not done:
                                if current_step == 1:
                                    # Force every beam's next node to be `head` by feeding a
                                    # one-hot trans_probs. Beamsearch.advance's first-step
                                    # branch (prev_Ks empty) clamps beams 1..N-1 to -1e20,
                                    # so all top-k results come from beam 0; with the one-hot
                                    # they all land on `head`. After this advance, all beams
                                    # share the same 2-step history [tail, head].
                                    trans_probs = torch.full(
                                        (batch_sub, beam_size, len_sub),
                                        -1e20,
                                        device=device,
                                    ).type(self.dtypeFloat)
                                    # head_be: (batch_sub, beam_size) -> scatter into last dim
                                    trans_probs.scatter_(2, head_be.unsqueeze(-1), 0.0)
                                else:
                                    # Distance-only scoring: -d(current_node, *).
                                    current_nodes = beamsearch.next_nodes[-1]
                                    dis_4d = dis_matrix.unsqueeze(1).expand(
                                        batch_sub, beam_size, len_sub, len_sub
                                    )
                                    idx = (
                                        current_nodes.unsqueeze(-1)
                                        .unsqueeze(-1)
                                        .expand(batch_sub, beam_size, 1, len_sub)
                                    )
                                    trans_probs = -dis_4d.gather(2, idx).squeeze(2)

                                self.env.selected_node_list = beamsearch.advance(
                                    trans_probs, self.env.selected_node_list
                                )
                                selected = beamsearch.next_nodes[-1].view(-1)
                                state, reward, done = self.env.step_beam(
                                    selected, beam=beam_size
                                )
                                current_step += 1

                            # Best beam per sub-instance. The beam can emit
                            # node-revisiting (non-permutation) hypotheses, whose
                            # measured length is bogusly small and would corrupt the
                            # global tour if selected. Two ways this happens:
                            #  - beam_size > len_sub: topk pads the beam with masked
                            #    candidates that duplicate already-visited nodes;
                            #  - coincident points (clustered/explosion data): the
                            #    multiplicative logit mask (score+(-dist))*1e20 fails
                            #    to suppress a visited node when dist==0 and score==0,
                            #    so it gets re-selected.
                            # Filter on actual permutation validity (every node
                            # 0..len_sub-1 visited exactly once); if no beam is valid,
                            # best_len is +inf and the segment is left unrepaired.
                            reward = reward.view(batch_sub, beam_size)
                            selected_node_list_be = self.env.selected_node_list.view(
                                batch_sub, beam_size, -1
                            )
                            sorted_nodes, _ = selected_node_list_be.sort(dim=2)
                            node_range = torch.arange(len_sub, device=device).view(
                                1, 1, -1
                            )
                            valid_beam = (sorted_nodes == node_range).all(dim=2)
                            reward = reward.masked_fill(~valid_beam, float("inf"))
                            best_len, min_idx = reward.min(dim=1)
                            zero_to_bsz = torch.arange(batch_sub, device=device)
                            best_sub_tour = selected_node_list_be[zero_to_bsz, min_idx]

                            ahter_repair_sub_solution = torch.roll(
                                best_sub_tour, shifts=-1, dims=1
                            )
                            after_repair_reward = best_len

                        after_repair_complete_solution = (
                            self.decide_whether_to_repair_solution(
                                ahter_repair_sub_solution,
                                partial_solution_length,
                                after_repair_reward,
                                indices,
                                len_of_sub,
                                select_node_list.clone(),
                            )
                        )

                        best_select_node_list = after_repair_complete_solution[:]

                        current_best_length = self.env._get_travel_distance_2(
                            origin_problem, best_select_node_list
                        )

            print("Get first complete solution!")
            escape_time, _ = clock.get_est_string(1, 1)
            gap = (
                (current_best_length.mean() - self.optimal_length.mean())
                / self.optimal_length.mean()
            ).item() * 100
            self.logger.info(
                "greedy, name:{}, gap:{:4f} %,  Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
                    name,
                    gap,
                    escape_time,
                    current_best_length.mean().item(),
                    self.optimal_length.mean().item(),
                )
            )

            ####################################################

            return (
                self.optimal_length.mean().item(),
                current_best_length.mean().item(),
                self.env.problem_size,
            )

    def sampling_subpaths_p(self, problems, solution, indices, len_of_sub):
        batch_size, problems_size, embedding_size = problems.shape
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        new_sulution = solution[:, indices]
        new_sulution_ascending, rank = torch.sort(
            new_sulution, dim=-1, descending=False
        )
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        index_2, _ = new_sulution_ascending.type(torch.long).sort(
            dim=-1, descending=False
        )
        index_2 = index_2.view(batch_size, -1)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(
            batch_size, index_2.shape[1]
        )
        new_data = problems[index_1, index_2].view(batch_size, -1, len_of_sub, 2)
        return new_data, new_sulution_rank
