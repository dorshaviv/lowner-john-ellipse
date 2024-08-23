import numpy as np
from matplotlib import pyplot as plt


##################
# main functions #
##################


def welzl(interior, boundary=np.zeros((0, 2))):
    """
    Find the smallest ellipse containing a set of points in the interior and
    another set of points on its boundary. To find the smallest ellipse
    containing a set of points without giben boundary points, the function can
    be called with the second argument empty (default usage).

    Arguments:
        interior: an array containing points in R2 as row vectors, representing
            points to be contained in the desired ellipse.
        boundary: an array containing points in R2 as row vectors, representing
            points to be on the boundary of the desired ellipse.

    Returns:
        an ellipse given by a tuple (c, a, b, t), where c = (x, y) is the
            center, a and b are the major and minor radii, and t is the
            rotation angle.
    """

    # stopping condition: stop either if interior is empty or if there are 5
    # points on the boundary, in which case a unique ellipse is determined.
    if interior.shape[0] == 0 or boundary.shape[0] >= 5:

        # if boundary has only 2 points, ellipse is degenerate
        if boundary.shape[0] <= 2:
            return None

        # call primitive functions to compute smallest ellipse going through
        # 3, 4, or 5 points.
        elif boundary.shape[0] == 3:
            return ellipse_from_boundary3(boundary)
        elif boundary.shape[0] == 4:
            return ellipse_from_boundary4(boundary)
        else:
            return ellipse_from_boundary5(boundary)

    # choose a random point in the interior set
    i = np.random.randint(interior.shape[0])
    p = interior[i, :]

    # remove point from interior set
    interior_wo_p = np.delete(interior, i, 0)

    # recursively call the function to find the smallest ellipse containing
    # the interior points without p
    ellipse = welzl(interior_wo_p, boundary)

    # if p is in this ellipse, then this ellipse is also the smallest ellipse
    # containing interior
    if is_in_ellipse(p, ellipse):
        return ellipse

    # if not, then p must be on the boundary of the smallest ellipse
    else:
        return welzl(interior_wo_p, np.vstack([boundary, p]))


##########################################################
# primitive functions used by Welzl recursion base cases #
##########################################################


def ellipse_from_boundary5(S):
    """
    Compute the unique ellipse that passes through 5 boundary points.

    Arguments:
        S: an array of shape (5,2) containing points in R2 as row
            vectors, which are on the boundary of the desired ellipse.

    Returns:
        an ellipse given by a tuple (c, a, b, t), where c = (x, y) is the
            center, a and b are the major and minor radii, and t is the
            rotation angle.
    """

    # find parameters of ellipse given in the form
    # s0 * x ** 2 + s1 * y ** 2 + 2 * s2 * x * y + s3 * x + s4 * y + 1 = 0.

    # build linear system of equations:
    x = S[:, 0]
    y = S[:, 1]
    A = np.column_stack((x**2, y**2, 2 * x * y, x, y))

    # if A is close to singular, then at least 3 points are colinear, in which
    # case an ellipse is not unique, then we give up on this ellipse
    if np.linalg.cond(A) >= 1 / np.finfo(float).eps:
        return None

    # solve system of equations
    sol = np.linalg.solve(A, -np.ones(S.shape[0]))

    # find ellipse center
    c = np.linalg.solve(-2 * np.array([[sol[0], sol[2]], [sol[2], sol[1]]]), sol[3:5])

    # solve for the matrix F (ellipse representation in center form)
    A = np.vstack(
        [
            np.hstack([np.eye(3), -np.array([[sol[0], sol[2], sol[1]]]).T]),
            np.array([c[0] ** 2, 2 * c[0] * c[1], c[1] ** 2, -1]),
        ]
    )
    s = np.linalg.solve(A, np.array([0, 0, 0, 1]))
    F = np.array([[s[0], s[1]], [s[1], s[2]]])

    return center_form_to_geometric(F, c)


def ellipse_from_boundary4(S):
    """
    Compute the smallest ellipse that passes through 4 boundary points,
    based on the algorithm by:
    B. W. Silverman and D. M. Titterington, "Minimum covering ellipses,"
    SIAM Journal on Scientific and Statistical Computing 1, no. 4 (1980):
    401-409.

    Arguments:
        S: an array of shape (4,2) containing points in R2 as row
            vectors, which are on the boundary of the desired ellipse.

    Returns:
        an ellipse given by a tuple (c, a, b, t), where c = (x, y) is the
            center, a and b are the major and minor radii, and t is the
            rotation angle. This ellipse is the ellipse with the smallest
            area that passes through the 4 points.
    """

    # sort coordinates in clockwise order
    Sc = S - np.mean(S, axis=0)
    angles = np.arctan2(Sc[:, 1], Sc[:, 0])
    S = S[np.argsort(-angles), :]

    # find intersection point of diagonals
    A = np.column_stack([S[2, :] - S[0, :], S[1, :] - S[3, :]])
    b = S[1, :] - S[0, :]
    s = np.linalg.solve(A, b)
    diag_intersect = S[0, :] + s[0] * (S[2, :] - S[0, :])

    # shift to origin
    S = S - diag_intersect

    # rotate so one diagonal is parallel to x-axis
    AC = S[2, :] - S[0, :]
    theta = np.arctan2(AC[1], AC[0])
    rot_mat = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    S = rot_mat.dot(S.T).T

    # shear parallel to x-axis to make diagonals perpendicular
    m = (S[1, 0] - S[3, 0]) / (S[3, 1] - S[1, 1])
    shear_mat = np.array([[1, m], [0, 1]], dtype=np.float)
    S = shear_mat.dot(S.T).T

    # make the quadrilateral cyclic (i.e. all vertices lie on a circle)
    b = np.linalg.norm(S, axis=1)
    d = b[1] * b[3] / (b[2] * b[0])
    stretch_mat = np.diag(np.array([d**0.25, d**-0.25], dtype=np.float))
    S = stretch_mat.dot(S.T).T

    # compute optimal swing angle by solving cubic equation
    a = np.linalg.norm(S, axis=1)
    coeff = np.zeros(4)
    coeff[0] = -4 * a[1] ** 2 * a[2] * a[0]
    coeff[1] = -4 * a[1] * (a[2] - a[0]) * (a[1] ** 2 - a[2] * a[0])
    coeff[2] = (
        3 * a[1] ** 2 * (a[1] ** 2 + a[2] ** 2)
        - 8 * a[1] ** 2 * a[2] * a[0]
        + 3 * (a[1] ** 2 + a[2] ** 2) * a[0] ** 2
    )
    coeff[3] = coeff[1] / 2.0
    rts = np.roots(coeff)
    # take the unique root in the interval (-1, 1)
    rts = rts[(-1 < rts) & (rts < 1)]
    theta = np.arcsin(np.real(rts[0]))

    # apply transformation D_theta
    D_mat = np.array(
        [
            [np.cos(theta) ** -0.5, np.sin(theta) * np.cos(theta) ** -0.5],
            [0, np.cos(theta) ** 0.5],
        ]
    )
    S = D_mat.dot(S.T).T

    # find enclosing circle
    boundary = S[:-1, :]  # only 3 points are needed
    A = np.vstack([-2 * boundary.T, np.ones(boundary.shape[0])]).T
    b = -np.sum(boundary**2, axis=1)
    s = np.linalg.solve(A, b)

    # circle parameters (center and radius)
    circle_c = s[:2]
    circle_r = np.sqrt(np.sum(circle_c**2) - s[2])

    # total affine transform that was applied
    T_mat = D_mat.dot(stretch_mat).dot(shear_mat).dot(rot_mat)

    # find original ellipse parameters (in center form)
    ellipse_c = np.linalg.solve(T_mat, circle_c) + diag_intersect
    ellipse_F = T_mat.T.dot(T_mat) / circle_r**2

    return center_form_to_geometric(ellipse_F, ellipse_c)


def ellipse_from_boundary3(S):
    """
    Compute the smallest ellipse that passes through 3 boundary points.

    Arguments:
        S: an array of shape (3,2) containing points in R2 as row
            vectors, which are on the boundary of the desired ellipse.

    Returns:
        an ellipse given by a tuple (c, a, b, t), where c = (x, y) is the
            center, a and b are the major and minor radii, and t is the
            rotation angle. This ellipse is the ellipse with the smallest
            area that passes through the 3 points.
    """

    # centroid
    c = np.mean(S, axis=0)

    # shift points
    Sc = S - c

    # ellipse matrix (center form)
    F = 1.5 * np.linalg.inv(Sc.T.dot(Sc))

    return center_form_to_geometric(F, c)


####################
# helper functions #
####################


def center_form_to_geometric(F, c):
    """
    Convert ellipse represented in center form:
        (x - c)^T * F * (x - c) = 1
    to geometrical representation, i.e. center, major-axis, minor-axis, and
    rotation angle.

    Arguments:
        F: array of shape (2,2), the matrix in the ellipse representation.
        c: array of length 2, the ellipse center.

    Returns:
        a tuple (c, a, b, t), where c = (x, y) is the center, a and
            b are the major and minor radii, and t is the rotation angle.
    """

    # extract a, b, and t from F by finding eigenvalues and eigenvectors
    w, V = np.linalg.eigh(F)

    # the eigenvalues are 1/a**2 and 1/b**2
    # the eigenvectors form the rotation matrix with angle t

    # if one the eigenvalues is not positive, the ellipse is degenerate
    if w[0] <= 0 or w[1] <= 0:
        return None

    # we assume without loss of generality 0 < t < pi.
    # V[1, 0] = sin(t), therefore it must be non-negative:
    if V[1, 0] < 0:
        V[:, 0] = -V[:, 0]

    # find t
    t = np.arccos(V[0, 0])  # V[0, 0] = cos(t)

    return c, 1 / np.sqrt(w[0]), 1 / np.sqrt(w[1]), t


def is_in_ellipse(point, ellipse):
    """
    Check if a point is contained in an ellipse.

    Arguments:
        point: array of length 2 representing a point in R2.
        ellipse: a tuple (c, a, b, t), where c = (x, y) is the center, a and
            b are the major and minor radii, and t is the rotation angle.

    Returns:
        bool: True if point is in ellipse, False otherwise.
    """

    # if ellipse is empty, return False
    if ellipse is None:
        return False

    # extract ellipse parameters
    c, a, b, t = ellipse

    # shift point by center of ellipse
    v = point - c

    # rotation matrix, by angle t
    rot_mat = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    # matrix F parametrizing ellipse in center form:
    # (x - c)^T * F * (x - c) = 1
    F = rot_mat.T.dot(np.diag(1 / np.array([a, b], dtype=np.float) ** 2)).dot(rot_mat)

    return v.T.dot(F.dot(v)) <= 1


#####################
# plotting function #
#####################


def sample_ellipse(ellipse, num_pts, endpoint=True):
    """
    Uniformly sample points on an ellipse.

    Arguments:
        ellipse: a tuple (c, a, b, t), where c = (x, y) is the center, a and
            b are the major and minor radii, and t is the rotation angle.
        num_pts: number of points to sample.
        endpoint: boolean. If True, repeat first point at the end (used for
            plotting).

    Returns:
        x: an array of shape (num_pts, 2) containing the sampled points as row
            vectors.
    """

    # extract ellipse parameters
    c, a, b, t = ellipse

    # rotation matrix
    rot_mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    # array of angles uniformly chosen between 0 and 2 * pi
    theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=endpoint)

    # points on an ellipse with axis a, b before rotation and shift
    z = np.column_stack((a * np.cos(theta), b * np.sin(theta)))

    # rotate points by angle t and shift to center c
    x = rot_mat.dot(z.T).T + c

    return x


def plot_ellipse(ellipse, num_pts=100, str="-"):
    """
    Plot ellipse.

    Arguments:
        ellipse: a tuple (c, a, b, t), where c = (x, y) is the center, a and
            b are the major and minor radii, and t is the rotation angle.
        num_pts: number of points to sample the ellipse and plot.
        str: plot string to be passed to plot function.
    """

    # if ellipse is empty, do nothing
    if ellipse is None:
        return

    # sample points on ellipse
    x = sample_ellipse(ellipse, num_pts)

    # plot ellipse
    plt.plot(x[:, 0], x[:, 1], str)


###############
# test script #
###############


def main():

    plt.figure()

    # generate random points in R2
    # points = np.random.randn(10, 2)
    points = np.array([[-1, 0], [0, 0], [1, 0], [0, 1]])
    plt.plot(points[:, 0], points[:, 1], ".")

    # find enclosing ellipse
    enclosing_ellipse = welzl(points)

    # plot resulting ellipse
    plot_ellipse(enclosing_ellipse, str="k--")

    plt.show()


if __name__ == "__main__":
    main()
