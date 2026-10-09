import sympy as sp
x=sp.Matrix(sp.symbols('x1:4',real=True)); v=sp.Matrix(sp.symbols('v1:4',real=True))
a,b=sp.symbols('a b',real=True)
b0=sp.Matrix([sp.Rational(2,7),sp.Rational(-3,5),sp.Rational(1,3)])
def EL_onshell(ell,f):
    out=[]
    for i in range(3):
        pv=sp.diff(ell,v[i])
        e=sum(sp.diff(pv,x[j])*v[j]+sp.diff(pv,v[j])*f[j] for j in range(3))-sp.diff(ell,x[i])
        out.append(e)
    return out
r2=x.dot(x); J=x.cross(v); xh_dot=(v*r2-x*x.dot(v))/r2**sp.Rational(3,2)   # d/dt x̂
w=xh_dot.dot(xh_dot)
cands={
 "l1 = |dx^/dt| (round sphere)": sp.sqrt(w),
 "l2 = (b.dx^/dt)^2/|dx^/dt| (Crofton, c~(b.n)^2)": (b0.dot(xh_dot))**2/sp.sqrt(w),
 "ctrl A: |x×v| (no 1/r^2)": sp.sqrt(J.dot(J)),
 "ctrl B: (b.dx^/dt)^4/|dx^/dt|^3": (b0.dot(xh_dot))**4/w**sp.Rational(3,2),
}
f_central=a*x+b*v          # any force in span(x,v): all central dynamics, incl. relativistic
f_noncentral=a*x+b*v+sp.Matrix([0,0,sp.Rational(1,2)])  # control field
import random; random.seed(1)
pts=[{**{x[i]:sp.Rational(random.randint(-9,9),7) for i in range(3)},**{v[i]:sp.Rational(random.randint(-9,9),11) for i in range(3)},a:sp.Rational(random.randint(-9,9),5),b:sp.Rational(random.randint(-9,9),3)} for _ in range(4)]
for name,ell in cands.items():
    for fn,f in [("central",f_central),("non-central",f_noncentral)]:
        E=EL_onshell(ell,f)
        vals=[max(abs(sp.N(sp.simplify(Ei.subs(p)),30)) for Ei in E) for p in pts]
        print(f"{name:48s} {fn:12s} max|EL| = {float(max(vals)):.3e}",flush=True)
