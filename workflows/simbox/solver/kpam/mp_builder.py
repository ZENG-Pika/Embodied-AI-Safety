import functools

import numpy as np
from solver.kpam import mp_terms, term_spec
from solver.kpam.optimization_problem import OptimizationProblemkPAM
from solver.kpam.transformations import affine_matrix_from_points


class OptimizationBuilderkPAM(object):
    def __init__(self, specification):
        self._specification = specification

    def _check_and_get_keypoint_idx(self, keypoint_name, input_idx=-1):
        # type: (str, int) -> int
        """
        Given the keypoint name, return the index for that keypoint.
        If input_idx is valid, also verify two index match each other
        :param keypoint_name:
        :param input_idx:
        :return:
        """
        assert keypoint_name in self._specification.keypoint_name2idx
        idx = self._specification.keypoint_name2idx[keypoint_name]

        if input_idx >= 0:
            assert idx == input_idx
        return idx

    def build_optimization(self, keypoint_observation):  # type: (np.ndarray) -> OptimizationProblemkPAM
        """
        Given the observed keypoint, build a mathematical program whose
        solution is the transform that map keypoint_observation to target
        :param keypoint_observation: (N, 3) keypoint location
        :return:
        """
        # Basic check
        assert keypoint_observation.shape[1] == 3
        assert keypoint_observation.shape[0] == len(self._specification.keypoint_name2idx)

        # Empty initialization of optimization problem
        problem = OptimizationProblemkPAM()
        problem.T_init = np.eye(4)
        problem.mp = None
        problem.xyzrpy = None
        problem.has_solution = False

        # Build the initial transformation
        problem.T_init = self._build_initial_transformation(keypoint_observation)

        # Build the mathematical_program with decision variable
        problem.build_empty_mp()

        # Process the cost terms
        for cost in self._specification.cost_list:
            self._process_cost_term(problem, keypoint_observation, cost)

        # Process the constraint terms
        for constraint in self._specification.constraint_list:
            self._process_constraint_term(problem, keypoint_observation, constraint)

        return problem

    def _build_initial_transformation(self, keypoint_observation):  # type: (np.ndarray) -> np.ndarray
        # The container for all keypoint that has nominal target
        from_keypoint = []
        to_keypoint = []

        # Get all the keypoint that has nominal target
        keypoint_name2idx = self._specification.keypoint_name2idx
        for keypoint_name in keypoint_name2idx:
            exist_target, location = self._specification.get_nominal_target_position(keypoint_name)
            if exist_target:
                to_keypoint.append(location)
                idx = keypoint_name2idx[keypoint_name]
                from_keypoint.append(
                    [
                        keypoint_observation[idx, 0],
                        keypoint_observation[idx, 1],
                        keypoint_observation[idx, 2],
                    ]
                )

        # Depends on whether we have enough keypoints
        assert len(from_keypoint) == len(to_keypoint)
        if len(from_keypoint) < 3:
            return np.eye(4)

        # There exists enough keypoint, let's compute using karbas algorithm
        # The np array are (3, N) format of keypoints
        from_keypoint_np = np.zeros(shape=(3, len(from_keypoint)))
        to_keypoint_np = np.zeros_like(from_keypoint_np)
        for i in range(len(from_keypoint)):
            for j in range(3):
                from_keypoint_np[j, i] = from_keypoint[i][j]
                to_keypoint_np[j, i] = to_keypoint[i][j]
        T = affine_matrix_from_points(from_keypoint_np, to_keypoint_np, shear=False, scale=False, usesvd=True)
        return T

    def _process_cost_term(self, problem, keypoint_observation, cost):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.OptimizationTermSpec) -> None
        """
        Global dispatcher for the cost term
        :param problem:
        :param cost:
        :return:
        """
        if isinstance(cost, term_spec.Point2PointCostL2Spec):
            self._process_keypoint_l2_cost_term(problem, keypoint_observation, cost)
        elif isinstance(cost, term_spec.Point2PlaneCostSpec):
            self._process_point2plane_cost_term(problem, keypoint_observation, cost)
        else:
            raise RuntimeError("The cost term type is unknown/not implemented")

    def _process_constraint_term(self, problem, keypoint_observation, constraint):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.OptimizationTermSpec) -> None
        """
        Global dispatcher for constraint term
        :param problem:
        :param keypoint_observation:
        :param constraint:
        :return:
        """
        if isinstance(constraint, term_spec.Point2PointConstraintSpec):
            self._process_point2point_constraint_term(problem, keypoint_observation, constraint)
        elif isinstance(constraint, term_spec.AxisAlignmentConstraintSpec):
            self._process_axis_alignment_constraint_term(problem, constraint)
        elif isinstance(constraint, term_spec.KeypointAxisAlignmentConstraintSpec):
            self._process_keypoint_axisalign_constraint_term(problem, keypoint_observation, constraint)
        elif isinstance(constraint, term_spec.KeypointAxisOrthogonalConstraintSpec):
            self._process_keypoint_axisorthogonal_constraint_term(problem, keypoint_observation, constraint)
        else:
            raise RuntimeError("The constraint type is unknown/not implemented")

    def _process_keypoint_l2_cost_term(self, problem, keypoint_observation, cost):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.Point2PointCostL2Spec) -> None
        idx = self._check_and_get_keypoint_idx(cost.keypoint_name, cost.keypoint_idx)

        # The current location
        from_keypoints = keypoint_observation[idx, :]
        from_keypoints = np.reshape(from_keypoints, (1, 3))

        # The target location
        to_keypoints = np.zeros(shape=(1, 3))
        to_keypoints[0, 0] = cost.target_position[0]
        to_keypoints[0, 1] = cost.target_position[1]
        to_keypoints[0, 2] = cost.target_position[2]

        # The cost function
        # It is wired that the cost term cannot use symbolic expression
        cost_func = functools.partial(
            mp_terms.keypoint_l2_cost,
            problem,
            from_keypoints,
            to_keypoints,
            cost.penalty_weight,
        )
        problem.mp.AddCost(cost_func, problem.xyzrpy)

    def _process_point2plane_cost_term(self, problem, keypoint_observation, cost):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.Point2PlaneCostSpec) -> None
        idx = self._check_and_get_keypoint_idx(cost.keypoint_name, cost.keypoint_idx)

        # The current location
        from_keypoints = keypoint_observation[idx, :]
        from_keypoints = np.reshape(from_keypoints, (1, 3))

        # The target location
        to_keypoints = np.zeros(shape=(1, 3))
        to_keypoints[0, 0] = cost.target_position[0]
        to_keypoints[0, 1] = cost.target_position[1]
        to_keypoints[0, 2] = cost.target_position[2]

        # The cost function
        # It is wired that the cost term cannot use symbolic expression
        cost_func = functools.partial(
            mp_terms.point2plane_cost,
            problem,
            from_keypoints,
            to_keypoints,
            cost.penalty_weight,
            cost.plane_normal,
        )
        problem.mp.AddCost(cost_func, problem.xyzrpy)

    def _process_point2point_constraint_term(self, problem, keypoint_observation, constraint):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.Point2PointConstraintSpec) -> None
        idx = self._check_and_get_keypoint_idx(constraint.keypoint_name, constraint.keypoint_idx)

        # The current location
        from_keypoint = keypoint_observation[idx, :]
        transformed_point_symbolic_val = mp_terms.transform_point_symbolic(problem, from_keypoint, problem.xyzrpy)

        # The target location
        to_keypoints = np.asarray(constraint.target_position).copy()
        to_keypoints_lb = to_keypoints - constraint.tolerance
        to_keypoints_ub = to_keypoints + constraint.tolerance
        for j in range(3):
            problem.mp.AddConstraint(transformed_point_symbolic_val[j] >= to_keypoints_lb[j])
            problem.mp.AddConstraint(transformed_point_symbolic_val[j] <= to_keypoints_ub[j])

    @staticmethod
    def _process_axis_alignment_constraint_term(problem, constraint):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.AxisAlignmentConstraintSpec) -> None
        assert constraint.tolerance < 1
        assert constraint.tolerance > 0

        # Check the axis
        from_axis = np.asarray(constraint.from_axis, dtype=np.float32).copy()  # type: np.ndarray
        to_axis = np.asarray(constraint.target_axis, dtype=np.float32).copy()  # type: np.ndarray
        norm_from = np.linalg.norm(from_axis)
        norm_to = np.linalg.norm(to_axis)
        if norm_from < 1e-4 or norm_to < 1e-4:
            print("Warning: the axis is zero, the constraint is not added to optimization.")
            return

        # Do normalization
        from_axis *= 1.0 / norm_from
        to_axis *= 1.0 / norm_to

        # Build the term
        vector_dot_symbolic_val = mp_terms.vector_dot_symbolic(problem, from_axis, to_axis, problem.xyzrpy)

        # Add to mp
        problem.mp.AddConstraint(
            vector_dot_symbolic_val >= constraint.target_inner_product - constraint.tolerance
        )  # - constraint.target_inner_product   # ← One space after #
        problem.mp.AddConstraint(
            vector_dot_symbolic_val <= constraint.target_inner_product + constraint.tolerance
        )  # - constraint.target_inner_product   # ← One space after #

    def _process_keypoint_axisalign_constraint_term(self, problem, keypoint_observation, constraint):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.KeypointAxisAlignmentConstraintSpec) -> None
        from_idx = self._check_and_get_keypoint_idx(
            constraint.axis_from_keypoint_name, constraint.axis_from_keypoint_idx
        )
        to_idx = self._check_and_get_keypoint_idx(constraint.axis_to_keypoint_name, constraint.axis_to_keypoint_idx)

        # Compute the axis
        from_point = keypoint_observation[from_idx, :]
        to_point = keypoint_observation[to_idx, :]
        vector = to_point - from_point
        norm_vec = np.linalg.norm(vector)
        if norm_vec < 1e-4:
            print("Warning: the axis is zero, the constraint is not added to optimization.")
            return
        vector *= 1.0 / norm_vec

        # Build axis alignment constraint
        axis_constraint = term_spec.AxisAlignmentConstraintSpec()
        axis_constraint.from_axis = [vector[0], vector[1], vector[2]]
        axis_constraint.target_axis = constraint.target_axis
        axis_constraint.tolerance = constraint.tolerance
        axis_constraint.target_inner_product = constraint.target_inner_product

        # OK
        self._process_axis_alignment_constraint_term(problem, axis_constraint)

    @staticmethod
    def _process_axis_orthogonal_constraint_term(problem, constraint):
        # make the axis orthogonal as the constraint
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.AxisAlignmentConstraintSpec) -> None
        assert constraint.tolerance < 1
        assert constraint.tolerance > 0

        # Check the axis
        from_axis = np.asarray(constraint.from_axis, dtype=np.float32).copy()  # type: np.ndarray
        to_axis = np.asarray(constraint.target_axis, dtype=np.float32).copy()  # type: np.ndarray
        norm_from = np.linalg.norm(from_axis)
        norm_to = np.linalg.norm(to_axis)
        if norm_from < 1e-4 or norm_to < 1e-4:
            print("Warning: the axis is zero, the constraint is not added to optimization.")
            return

        # Do normalization
        from_axis *= 1.0 / norm_from
        to_axis *= 1.0 / norm_to

        # Build the term
        vector_dot_symbolic_val = mp_terms.vector_dot_symbolic(problem, from_axis, to_axis, problem.xyzrpy)

        # Add to mp
        problem.mp.AddConstraint(
            vector_dot_symbolic_val - constraint.target_inner_product <= constraint.tolerance
        )  # 1.0 -
        problem.mp.AddConstraint(
            vector_dot_symbolic_val - constraint.target_inner_product >= -constraint.tolerance
        )  # 1.0 -

    def _process_keypoint_axisorthogonal_constraint_term(self, problem, keypoint_observation, constraint):
        # type: (OptimizationProblemkPAM, np.ndarray, term_spec.KeypointAxisAlignmentConstraintSpec) -> None
        from_idx = self._check_and_get_keypoint_idx(
            constraint.axis_from_keypoint_name, constraint.axis_from_keypoint_idx
        )
        to_idx = self._check_and_get_keypoint_idx(constraint.axis_to_keypoint_name, constraint.axis_to_keypoint_idx)

        # Compute the axis
        from_point = keypoint_observation[from_idx, :]
        to_point = keypoint_observation[to_idx, :]
        vector = to_point - from_point
        norm_vec = np.linalg.norm(vector)
        if norm_vec < 1e-4:
            print("Warning: the axis is zero, the constraint is not added to optimization.")
            return
        vector *= 1.0 / norm_vec

        # Build axis alignment constraint
        axis_constraint = term_spec.AxisAlignmentConstraintSpec()
        axis_constraint.from_axis = [vector[0], vector[1], vector[2]]
        axis_constraint.target_axis = constraint.target_axis
        axis_constraint.tolerance = constraint.tolerance
        axis_constraint.target_inner_product = constraint.target_inner_product

        # OK
        self._process_axis_orthogonal_constraint_term(problem, axis_constraint)


def test_builder():
    """Legacy test entry; no longer used."""
    raise RuntimeError("test_builder is deprecated and should not be called.")
