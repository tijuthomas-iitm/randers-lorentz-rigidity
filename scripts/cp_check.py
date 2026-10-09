import sympy as sp, random
x=sp.Matrix(sp.symbols('x1:4',real=True)); v=sp.Matrix(sp.symbols('v1:4',real=True)); a,b=sp.symbols('a b',real=True)
r=sp.sqrt(x.dot(x)); J=sp.sqrt(x.cross(v).dot(x.cross(v)))
def EL(ell,f):
    return [sum(sp.diff(sp.diff(ell,v[i]),x[j])*v[j]+sp.diff(sp.diff(ell,v[i]),v[j])*f[j] for j in range(3))-sp.diff(ell,x[i]) for i in range(3)]
random.seed(3)
pts=[{**{x[i]:sp.Rational(random.randint(-9,9),7) for i in range(3)},**{v[i]:sp.Rational(random.randint(-9,9),11) for i in range(3)},a:sp.Rational(random.randint(-9,9),5),b:sp.Rational(random.randint(-9,9),3)} for _ in range(3)]
for name,ell in [("J/r^2 (ours)",J/r**2),("J/r (Crampin-Prince, as printed)",J/r),("J",J)]:
    for fn,f in [("f=a x (Newtonian central)",a*x),("f=a x+b v",a*x+b*v)]:
        E=EL(ell,f); print(f"{name:34s} {fn:26s} max|EL|={max(float(abs(sp.N(Ei.subs(p),30))) for Ei in E for p in pts):.3e}",flush=True)
