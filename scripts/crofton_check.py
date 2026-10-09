import sympy as sp, random
th=sp.symbols('th',real=True)
x=sp.Matrix(sp.symbols('x1:4',real=True)); v=sp.Matrix(sp.symbols('v1:4',real=True)); a,b_=sp.symbols('a b',real=True)
r2=x.dot(x); xh=x/sp.sqrt(r2); W=(v*r2-x*x.dot(v))/r2**sp.Rational(3,2); nw=sp.sqrt(W.dot(W))
B=sp.Matrix([sp.Rational(2,7),sp.Rational(-3,5),sp.Rational(1,3)]); Dv=sp.Matrix([sp.Rational(1,2),sp.Rational(1,4),sp.Rational(-2,3)])
# angular integral over n = cos th w^ + sin th m, weight |cos th|; use symmetry: 4*int_0^{pi/2} for even-in-sin terms
def I(k,l):  # int_0^{2pi} |cos|^{1} cos^k sin^l
    if l%2: return 0
    return 2*sp.integrate(sp.cos(th)**(k+1)*sp.sin(th)**l,(th,-sp.pi/2,sp.pi/2))*(1 if k%2==0 else 0)
def F_of(coeffs):  # c(n)=sum C[k,l] (n.w^)^k (n.m)^l ; returns |w| * sum C I
    return nw*sum(C*I(k,l) for (k,l),C in coeffs.items())
bw=B.dot(W)/nw; dw=Dv.dot(W)/nw
bm2=B.dot(B)-B.dot(xh)**2-bw**2; dm2=Dv.dot(Dv)-Dv.dot(xh)**2-dw**2; bmdm=B.dot(Dv)-B.dot(xh)*Dv.dot(xh)-bw*dw
# (b.n)^4 = (bw c + bm s)^4 ; only even powers of s survive
quart={(4,0):bw**4,(2,2):6*bw**2*bm2,(0,4):bm2**2}
# (b.n)^2 (d.n)^2 even-in-s part
mixed={(4,0):bw**2*dw**2,(2,2):bw**2*dm2+dw**2*bm2+4*bw*dw*bmdm,(0,4):bm2*dm2}
cands={"Crofton c=(b.n)^2(d.n)^2":F_of(mixed),
       "ctrl: (0,4) term doubled":F_of({**quart,(0,4):2*bm2**2})}
def EL(ell,f):
    return [sum(sp.diff(sp.diff(ell,v[i]),x[j])*v[j]+sp.diff(sp.diff(ell,v[i]),v[j])*f[j] for j in range(3))-sp.diff(ell,x[i]) for i in range(3)]
random.seed(7)
pts=[{**{x[i]:sp.Rational(random.randint(-9,9),7) for i in range(3)},**{v[i]:sp.Rational(random.randint(-9,9),11) for i in range(3)},a:sp.Rational(random.randint(-9,9),5),b_:sp.Rational(random.randint(-9,9),3)} for _ in range(3)]
for nm,ell in cands.items():
    E=EL(ell,a*x+b_*v)
    print(nm, max(float(abs(sp.N(Ei.subs(p),40))) for Ei in E for p in pts),flush=True)
