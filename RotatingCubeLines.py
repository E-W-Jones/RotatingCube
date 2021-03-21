# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:21:07 2021.
Author: Evan Jones
"""
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class RotationMatrix():
    """A class of rotation matrices, X, Y, Z refers to the axis being rotated about."""
    def X(theta):
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, cos(theta), -sin(theta)],
                         [0.0, sin(theta), cos(theta)]])

    def Y(theta):
        return np.array([[cos(theta), 0.0, sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-sin(theta), 0.0, cos(theta)]])

    def Z(theta):
        return np.array([[cos(theta), -sin(theta), 0],
                         [sin(theta), cos(theta), 0],
                         [0, 0, 1]])


class RotatingCube():
    def __init__(self, centre, side_length, radius=0.1):
        """
        Initialise the cube.

        Parameters
        ----------
        centre : Three dimension vector (list or array) of the mid-point
        side_length : Float of the length of the sides
        radius : Radius of the points that mark the corners

        """
        self.mid = np.array(centre, dtype=float)
        self.L = float(side_length)
        # Generate x,y,z unit vectors
        x, y, z = np.eye(3, dtype=float)
        l = self.L / 2
        self.corners = np.vstack((self.mid + l * (x + y + z),
                                  self.mid + l * (x - y + z),
                                  self.mid + l * (- x - y + z),
                                  self.mid + l * (- x + y + z),
                                  self.mid + l * (x + y - z),
                                  self.mid + l * (x - y - z),
                                  self.mid + l * (- x - y - z),
                                  self.mid + l * (- x + y - z)))
        projection = self.project()
        # I'm sure this could be made a lot neater
        self.line_list = [plt.Line2D(projection[:2, 0], projection[:2, 1], color='tab:blue'),
                          plt.Line2D(projection[1:3, 0], projection[1:3, 1], color='tab:blue'),
                          plt.Line2D(projection[2:4, 0], projection[2:4, 1], color='tab:blue'),
                          plt.Line2D(projection[:4:3, 0], projection[0:4:3, 1], color='tab:blue'),
                          plt.Line2D(projection[4:6, 0], projection[4:6, 1], color='tab:orange'),
                          plt.Line2D(projection[5:7, 0], projection[5:7, 1], color='tab:orange'),
                          plt.Line2D(projection[6:, 0], projection[6:, 1], color='tab:orange'),
                          plt.Line2D(projection[4::3, 0], projection[4::3, 1], color='tab:orange'),
                          plt.Line2D(projection[0:5:4, 0], projection[0:5:4, 1], color='tab:green'),
                          plt.Line2D(projection[1:6:4, 0], projection[1:6:4, 1], color='tab:green'),
                          plt.Line2D(projection[2:7:4, 0], projection[2:7:4, 1], color='tab:green'),
                          plt.Line2D(projection[3:8:4, 0], projection[3:8:4, 1], color='tab:green')]

    def rotate(self, angles=[0, 0, 0], order="XYZ"):
        """
        Rotate each corner around all axes by an angle.

        Parameters
        ----------
        angles : A list of the angles the cube is rotated by, in radians
        order : Which axis is rotated around first, second, then third

        """
        matrices = {"X": RotationMatrix.X,
                    "Y": RotationMatrix.Y,
                    "Z": RotationMatrix.Z}
        
        R = np.eye(3)
        for theta, axis in zip(angles, order):
            R = matrices[axis](theta) @ R

        # A fancier way to rotate all the vectors in one function call,
        # as opposed to using loops
        self.corners = np.einsum('jk, ik -> ij', R, self.corners)

    def orthoraphic_projection(self):
        """
        Project the points onto the x-y plane orthographically.

        Returns
        -------
        An array of the corners without the z-component

        """
        return self.corners[:, :2]

    def perspective_projection(self):
        """
        Project the points onto the x-y plane with perspective.

        Returns
        -------
        An array of the corners projected into x-y plane

        """
        # I don't really understand the terms and formulae, just from wikipedia
        # Position of the camera
        c = np.array([0, 0, 8], dtype=float)
        d = self.corners - c
        # Position of the image plane
        ip = np.array([0, 0, 11])
        vector_list = [[ip[2] * di[0] / di[2] + ip[0], ip[2] * di[1] / di[2] + ip[1]] for di in d]
        return np.array(vector_list)

    def project(self, kind="perspective"):
        """
        Return the projection of the cube's corners into the x-y plane.

        Offers either orthographic or perspective projections.

        Parameters
        ----------
        kind : The kind of projection, either "perspective" (default) or "orthographic".

        Returns
        --------
        The projections of the corners as an 8x2 numpy array.
        """
        if kind == "perspective":
            return self.perspective_projection()
        else:
            return self.orthoraphic_projection()

    def anim_init(self):
        """Initialiser for FuncAnimation."""
        return self.line_list

    def animate(self, i):
        """
        Step forward the animation, called by FuncAnimation.

        Rotates and then projects the corners into the x-y plane,
        before updating the coordinates for the lines.

        Parameters
        ----------
        i : Doesn't mean anything, it's just there to work with FuncAnimation.

        Returns
        -------
        A list of all the lines to be animated.
        """
        self.rotate()
        projection = self.project()
        new_points = [(projection[:2, 0], projection[:2, 1]),
                    (projection[1:3, 0], projection[1:3, 1]),
                    (projection[2:4, 0], projection[2:4, 1]),
                    (projection[:4:3, 0], projection[0:4:3, 1]),
                    (projection[4:6, 0], projection[4:6, 1]),
                    (projection[5:7, 0], projection[5:7, 1]),
                    (projection[6:, 0], projection[6:, 1]),
                    (projection[4::3, 0], projection[4::3, 1]),
                    (projection[0:5:4, 0], projection[0:5:4, 1]),
                    (projection[1:6:4, 0], projection[1:6:4, 1]),
                    (projection[2:7:4, 0], projection[2:7:4, 1]),
                    (projection[3:8:4, 0], projection[3:8:4, 1])]
        for line, points in zip(self.line_list, new_points):
            line.set_data(points)
        return self.line_list

    def run(self, show=True, save=False):
        """
        Run the animation of a rotating cube.

        Currently has a lot of things hardcoded that might be better as arguments.

        Parameters
        ----------
        show : Boolean, if True (default) it shows the animation, if false it doesn't.
        save : Boolean, if True it saves the animation, if false (default) it doesn't.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        x_width = self.mid[0] + self.L
        y_width = self.mid[1] + self.L
        ax.set_xlim(-x_width, x_width)
        ax.set_ylim(-y_width, y_width)
        for line in self.line_list:
            ax.add_line(line)

        anim = FuncAnimation(fig, self.animate, init_func=self.anim_init,
                             interval=1/30, blit=True, frames=150)
        if show:
            plt.show()
        if save:
            anim.save("RotatingCube.gif")

def main():
    cube = RotatingCube([0, 0, 0], 1)
    cube.run()


if __name__ == "__main__":
    main()
