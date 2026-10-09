# Verify: for F = b(t,x) dt^dx (constant kernel span(e2,e3)), Lorentz trajectories are extremals of
# G~ = alpha + beta + eps*H(y2,y3), H = sqrt(y2^2 + 2 y3^2) (degree 1, nonlinear).  Units c=1, q/m=1.
import numpy as np
from scipy.integrate import solve_ivp
eta=np.diag([1.,-1,-1,-1])
def A(x):   # potential: A_mu (lower), A0 = phi(t,x1) -> F = dA has only t-x1 components
    t,x1=x[0],x[1]; return np.array([0.3*x1+0.2*x1**2*t+0.1*np.sin(x1+t),0,0,0])
def dA(x,h=1e-6):  # dA[mu][nu] = d_mu A_nu
    J=np.zeros((4,4))
    for m in range(4):
        e=np.zeros(4); e[m]=h; J[m]=(A(x+e)-A(x-e))/(2*h)
    return J
def Fdn(x): J=dA(x); return J-J.T          # F_{mu nu}
def rhs(tau,s):
    x,u=s[:4],s[4:]
    Fud=eta@Fdn(x)                          # F^mu_nu
    return np.concatenate([u,Fud@u])
u0=np.array([0,0.3,0.4,-0.2]); u0[0]=np.sqrt(1+u0[1:]@u0[1:])
sol=solve_ivp(rhs,[0,3],np.concatenate([[0,0.1,0.2,0.3],u0]),rtol=1e-11,atol=1e-12,dense_output=True)
eps=0.2
def G(x,y):
    a=np.sqrt(y@eta@y); return a+A(x)@y+eps*np.sqrt(y[2]**2+2*y[3]**2)
def EL(Gf,tau,h=1e-4,d=1e-6):
    def p(tt):   # dG/dy along curve
        s=sol.sol(tt); x,y=s[:4],s[4:]
        return np.array([(Gf(x,y+d*np.eye(4)[m])-Gf(x,y-d*np.eye(4)[m]))/(2*d) for m in range(4)])
    s=sol.sol(tau); x,y=s[:4],s[4:]
    dp=(p(tau+h)-p(tau-h))/(2*h)
    dx=np.array([(Gf(x+d*np.eye(4)[m],y)-Gf(x-d*np.eye(4)[m],y))/(2*d) for m in range(4)])
    return dp-dx
F0=lambda x,y: np.sqrt(y@eta@y)+A(x)@y
for tau in [0.5,1.5,2.5]:
    print(tau,"EL(F) =",np.abs(EL(F0,tau)).max().round(7)," EL(F+eps*H) =",np.abs(EL(G,tau)).max().round(7))
# control: H on a NON-kernel component (y1) should fail
Gbad=lambda x,y: F0(x,y)+eps*np.sqrt(y[1]**2+2*y[3]**2)
print("control (H uses y1, not in kernel):",np.abs(EL(Gbad,1.5)).max().round(5))
