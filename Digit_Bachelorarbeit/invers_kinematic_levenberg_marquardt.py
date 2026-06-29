import numpy as np
import jax.numpy as jnp
import jax

from kinematic.pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt_jax import (
    pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt,
)
from kinematic.pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt_jax import (
    pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt,
)
from kinematic.J_pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt_jax import (
    J_pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt,
)
from kinematic.J_pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt_jax import (
    J_pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt,
)


def legs_levenberg_marquardt(q_init, pos_des_left_toe, pos_des_right_toe):
    q_init = q_init.copy()
    q_left_lower_limit = np.array(
        [
            -1.0471975511965976,
            -0.6981317007977318,
            -1.0471975511965976,
            -1.3962634015954636,
            -0.767944870877505,
        ]
    )
    q_left_upper_limit = np.array(
        [
            1.0471975511965976,
            0.6981317007977318,
            1.5707963267948966,
            1.0192722831646885,
            0.5934119456780721,
        ]
    )

    q_right_lower_limit = np.array(
        [
            -1.0471975511965976,
            -0.6981317007977318,
            -1.5707963267948966,
            -1.0192722831646885,
            -0.5934119456780721,
        ]
    )

    q_right_upper_limit = np.array(
        [
            1.0471975511965976,
            0.6981317007977318,
            1.0471975511965976,
            1.3962634015954636,
            0.767944870877505,
        ]
    )
    # define the desired position left toe
    pos_des_left = pos_des_left_toe[:5]

    # define the desired position right toe
    pos_des_right = pos_des_right_toe[:5]

    # Define the initial guess for the damping factor lambda as a vector
    lambda_init = 1e-8  # 5e-2  # 1e-8

    lambda_diag = np.eye(5) * lambda_init

    q_left = np.zeros(5)
    q_left[0] = q_init[6]
    q_left[1] = q_init[7]
    q_left[2] = q_init[8]
    q_left[3] = q_init[9]
    q_left[4] = q_init[14]

    q_right = np.zeros(5)
    q_right[0] = q_init[16]
    q_right[1] = q_init[17]
    q_right[2] = q_init[18]
    q_right[3] = q_init[19]
    q_right[4] = q_init[24]

    q_left_full = np.zeros(6)
    q_right_full = np.zeros(6)

    for _ in range(10):
        # calc FK

        q_left_full[0] = q_left[0]
        q_left_full[1] = q_left[1]
        q_left_full[2] = q_left[2]
        q_left_full[3] = q_left[3]
        # q_left_full[4] = 0; #! can not be conntrolled
        # q_left_full[5] = -q_left[3] #! is set internally to -q_left[3]
        q_left_full[5] = q_left[4]
        #   q_left_full[6] = 0; #! should not be conntrolled

        q_right_full[0] = q_right[0]
        q_right_full[1] = q_right[1]
        q_right_full[2] = q_right[2]
        q_right_full[3] = q_right[3]
        # q_right_full[4] = 0; #! can not be conntrolled
        # q_right_full[5] = -q_right[3] #! is set internally to -q_right[3]
        q_right_full[5] = q_right[4]
        #   q_right_full[6] = 0; #! should not be conntrolled

        pos_left = pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt(q_left_full)[0:5]

        pos_right = pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt(q_right_full)[
            0:5
        ]

        # Compute the Jacobian
        Jm_l = J_pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt(q_left_full)[
            0:5, 0:5
        ]
        # remove last row and last column
        J_transpose_l = Jm_l.T

        Jm_r = J_pos_ypr_fk_base_link_right_toe_Levenberg_Marquardt(q_right_full)[
            0:5, 0:5
        ]
        J_transpose_r = Jm_r.T

        # calc Hessian
        hessian_approx_left = J_transpose_l @ Jm_l
        hessian_approx_right = J_transpose_r @ Jm_r

        # compute the error
        error_left = pos_des_left - pos_left
        error_right = pos_des_right - pos_right

        hessian_left = hessian_approx_left + lambda_diag + 0.5 * hessian_approx_left
        hessian_right = hessian_approx_right + lambda_diag + 0.5 * hessian_approx_right

        q_update_left = jnp.linalg.inv(hessian_left) @ J_transpose_l @ error_left
        q_update_right = jnp.linalg.inv(hessian_right) @ J_transpose_r @ error_right

        # update q
        q_left += q_update_left
        q_right += q_update_right

        # clip q
        q_left = np.clip(q_left, q_left_lower_limit, q_left_upper_limit)
        q_right = np.clip(q_right, q_right_lower_limit, q_right_upper_limit)

    q_init[6] = q_left[0]
    q_init[7] = q_left[1]
    q_init[8] = q_left[2]
    q_init[9] = q_left[3]
    q_init[11] = -q_left[3]
    q_init[14] = q_left[4]

    q_init[16] = q_right[0]
    q_init[17] = q_right[1]
    q_init[18] = q_right[2]
    q_init[19] = q_right[3]
    q_init[21] = -q_right[3]
    q_init[24] = q_right[4]

    return q_init


def main():
    q_init = np.zeros(34) #+ 1e-3

    pos_left_foot = np.array(
        [
            -0.045,
            0.154,
            -0.851,
            0.0,
            0.0,
        ]
    )

    q_right_foot = np.array(
        [
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
        ]
    )

    print("left_foot_des", pos_left_foot)
    print("old q", q_init)
    q_new = legs_levenberg_marquardt(q_init, pos_left_foot, q_right_foot)
    q_left_foot = np.array(
        [
            q_new[6],
            q_new[7],
            q_new[8],
            q_new[9],
            q_new[14],
        ]
    )
    print(q_left_foot)
    
    new_pos = pos_ypr_fk_base_link_left_toe_Levenberg_Marquardt(q_left_foot)#[:5]
    # error = pos_left_foot - new_pos
    

    print("new q", q_new)
    # print("error", error[:3], np.rad2deg(error[3:]))
    # print("error", error)

    print(q_left_foot)


if __name__ == "__main__":
    main()
