# Is there h(J1,J2,J3,E) with sum h_I EL(I) = 0 on shell?  Pointwise null vector c of [EL(J1..3),EL(E)] (3x4);
# necessary: c's direction is constant along each trajectory (depends only on integrals).
import sympy as sp, numpy as np
from scipy.integrate import solve_ivp
x=sp.symbols('x1:4'); v=sp.symbols('v1:4')
r=sp.sqrt(sum(q**2 for q in x)); Phi=-sp.Rational(1,2)/r     # Coulomb (attractive)
g=1/sp.sqrt(1-sum(q**2 for q in v)); E=[-sp.diff(Phi,xi) for xi in x]; vE=sum(v[i]*E[i] for i in range(3))
f=[(E[i]-v[i]*vE)/g for i in range(3)]
J=[g*(x[(a+1)%3]*v[(a+2)%3]-x[(a+2)%3]*v[(a+1)%3]) for a in range(3)]; En=g+Phi
def EL(I): return [sum(sp.diff(I,v[c],x[d])*v[d]+sp.diff(I,v[c],v[d])*f[d] for d in range(3))-sp.diff(I,x[c]) for c in range(3)]
M=sp.lambdify(x+v,sp.Matrix([EL(I) for I in J+[En]]).T)       # 3x4
F=sp.lambdify(x+v,f); IJ=sp.lambdify(x+v,J+[En])
s0=[0.8,0.1,0.3, 0.05,0.35,0.1]
sol=solve_ivp(lambda t,s: list(s[3:])+list(np.array(F(*s),float)),[0,2],s0,rtol=1e-9,atol=1e-10,method="DOP853",dense_output=True)
for t in [0,0.5,1,1.5,2]:
    s=sol.sol(t); m=np.array(M(*s),float); u,sv,vt=np.linalg.svd(m); c=vt[-1]; c=c/np.sign(c[np.argmax(abs(c))])/np.linalg.norm(c)
    print(f"t={t} integrals={np.round(IJ(*s),6)} sing.vals={np.round(sv,5)} null c={np.round(c,5)}")
