# Exact checks of identities added in the expanded derivations (manuscript Sections 2-7).
import sympy as sp
v=sp.Matrix(sp.symbols('v1:4',real=True)); s=sp.sqrt(1-(v.T*v)[0]); g=1/s
# (a) Hess of sqrt(1-|v|^2) = -gamma (I + gamma^2 v v^T)
H=sp.hessian(s,v); print("(a) Hessian identity residual:", sp.simplify(H+g*(sp.eye(3)+g**2*v*v.T)))
# (b) h + h'' = 2[c(phi+pi/2)+c(phi-pi/2)] for h(phi)=int |cos(th-phi)| c(th) dth, test c=1+cos(th)^2/2+sin(3th)/5 (positive)
th,ph=sp.symbols('theta phi',real=True)
c=lambda x: 1+sp.cos(x)**2/2+sp.sin(3*x)/5
import mpmath as mp
def h(p):
    f=lambda t: abs(mp.cos(t-p))*(1+mp.cos(t)**2/2+mp.sin(3*t)/5)
    pts=[p-mp.pi/2,p+mp.pi/2,p+3*mp.pi/2]
    return mp.quad(f,[p-mp.pi/2,p+mp.pi/2,p+3*mp.pi/2])
mp.mp.dps=30
for p in [mp.mpf('0.3'),mp.mpf('1.7')]:
    lhs=h(p)+mp.diff(h,p,2); rhs=2*(c(sp.Float(str(p+mp.pi/2),30))+c(sp.Float(str(p-mp.pi/2),30)))
    print("(b) h+h'' vs 2[c+c]:", mp.nstr(lhs,15), sp.N(rhs,15))
# (c) |d rhat/dt| = |r x v|/|r|^2
r=sp.Matrix(sp.symbols('r1:4',real=True)); rn=sp.sqrt((r.T*r)[0])
drh=(v-r*(r.T*v)[0]/rn**2)/rn
print("(c) speed identity residual:", sp.simplify((drh.T*drh)[0]-(r.cross(v).T*r.cross(v))[0]/rn**4))
# (d) EL of a0(t,x)+a(t,x).v = d_t a - grad a0 - v x curl a
t=sp.symbols('t'); X=sp.symbols('x1:4'); a0=sp.Function('a0')(t,*X); a=[sp.Function('a%d'%i)(t,*X) for i in (1,2,3)]
L=a0+sum(a[i]*v[i] for i in range(3))
EL=[sp.diff(sp.diff(L,v[i]),t)+sum(v[j]*sp.diff(sp.diff(L,v[i]),X[j]) for j in range(3))-sp.diff(L,X[i]) for i in range(3)]
curl=sp.Matrix([sp.diff(a[2],X[1])-sp.diff(a[1],X[2]),sp.diff(a[0],X[2])-sp.diff(a[2],X[0]),sp.diff(a[1],X[0])-sp.diff(a[0],X[1])])
rhs=[sp.diff(a[i],t)-sp.diff(a0,X[i])-v.cross(curl)[i] for i in range(3)]
print("(d) gauge EL residual:", [sp.simplify(EL[i]-rhs[i]) for i in range(3)])
# (e) uniform B along z: L = L_R - eps*sqrt(1+vz^2) has EL vanishing on Lorentz solutions
Bz,eps=sp.symbols('B eps'); 
f=s*sp.Matrix([v[1]*Bz,-v[0]*Bz,0])   # f = sqrt(1-v^2)(v x B), B=(0,0,Bz)
ell=sp.sqrt(1+v[2]**2)
ELell=[sum(sp.diff(ell,v[i],v[j])*f[j] for j in range(3)) for i in range(3)]  # x-independent
print("(e) EL of sqrt(1+vz^2) on uniform-B solutions:", [sp.simplify(e) for e in ELell])
