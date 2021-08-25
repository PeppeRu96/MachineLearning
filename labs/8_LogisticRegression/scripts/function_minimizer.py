import numpy as np
import scipy.optimize

def f(x):
    y = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2
    return y

def fWithGrad(x):
    y = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2
    dfX0 = 2 * (x[0] + 3) + np.cos(x[0])
    dfX1 = 2 * (x[1] + 1)
    grad = np.array([dfX0, dfX1])
    return (y, grad)

if __name__ == "__main__":
    print("Experiment with the numerical optimization problem through L_BFGS_B method..")

    starting_pt = np.array([0, 0])
    print("Calculating minimum of the function f with approximated gradients...")
    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(f, starting_pt, approx_grad=True, iprint=0)
    print("Number of f() calls: ", d['funcalls'])
    print("X min with approximated gradients: ", xMin)
    print("f(xMin) with approximated gradients: ", fMin)
    print("\n")

    print("Calculating minimum of the function f..")
    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(fWithGrad, starting_pt, iprint=0)
    print("Number of f() calls: ", d['funcalls'])
    print("X min: ", xMin)
    print("f(xMin): ", fMin)