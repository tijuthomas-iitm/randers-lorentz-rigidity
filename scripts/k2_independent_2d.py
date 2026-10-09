# K2: independent re-implementation (sympy Poly + flint), own field, f derived from L0 symbolically checked.
import sys, random, sympy as sp
from flint import fmpq_mat, fmpq
n=2; N=int(sys.argv[1]) if len(sys.argv)>1 else 3; D=N+3
t,x1,x2,w1,w2,eps=sp.symbols('t x1 x2 w1 w2 eps')
X=[x1,x2]; W=[w1,w2]; G=(t,x1,x2,w1,w2)
v0=[sp.Rational(3,13),sp.Rational(4,13)]   # 1-|v0|^2=(12/13)^2
V=[v0[i]+W[i] for i in range(2)]
rnd=random.Random(int(sys.argv[2]) if len(sys.argv)>2 else 99)
def rpoly(vars_,deg):
    return sum(sp.Rational(rnd.randint(-4,4),rnd.randint(1,3))*sp.prod([v**e for v,e in zip(vars_,ex)])
               for ex in sp.itermonomials.__wrapped__(vars_,deg) ) if False else \
           sum(sp.Rational(rnd.randint(-4,4),rnd.randint(1,3))*m for m in sp.itermonomials([t]+X,deg))
mode=sys.argv[3] if len(sys.argv)>3 else 'gen'
if mode=='gen':
    Phi=rpoly(0,3); A=[rpoly(0,3) for _ in range(2)]
elif mode=='radial':
    xc=[sp.Rational(1,3),sp.Rational(-1,3),sp.Rational(2,3)]
    Phi=sp.Rational(2,3)*sum((X[i]-xc[i])**2 for i in range(2)); A=[0,0,0]
else:
    Phi=0; A=[-sp.Rational(3,5)*x2, sp.Rational(3,5)*x1, 0]   # uniform B_z
P=lambda e: sp.Poly(sp.expand(e),*G,domain='QQ')
def tr(p,d): return sp.Poly.from_dict({m:c for m,c in p.as_dict().items() if sum(m)<=d},*G,domain='QQ')
# --- field and series
E=[-sp.diff(Phi,X[i])-sp.diff(A[i],t) for i in range(2)]
Bz=sp.diff(A[1],x1)-sp.diff(A[0],x2)
sexpr=sp.sqrt(1-sum((v0[i]+eps*W[i])**2 for i in range(2)))
sser=sp.series(sexpr,eps,0,D+1).removeO().subs(eps,1)
S=P(sser)
vE=sum(V[k]*E[k] for k in range(2))
cr=[V[1]*Bz,-V[0]*Bz]
f=[tr(S*P(E[i]+cr[i]-V[i]*vE),D) for i in range(2)]
d=lambda p,var: p.diff(var)
def Gam(p):
    r=d(p,t)
    for a in range(2): r=r+P(V[a])*d(p,X[a])+f[a]*d(p,W[a])
    return tr(r,D)
Gm=[[tr(-sp.Rational(1,2)*d(f[i],W[j]),D) for j in range(2)] for i in range(2)]       # Gamma^i_j
Ph=[[tr(-d(f[i],X[j]) - sum((Gm[i][k]*Gm[k][j] for k in range(2)),P(0)) - Gam(Gm[i][j]),D) for j in range(2)] for i in range(2)]
pairs=[(i,j) for i in range(2) for j in range(i,2)]
def resid(g):
    out=[]
    for (i,j) in pairs:
        r=Gam(g[i][j]) - sum((g[i][k]*Gm[k][j]+g[j][k]*Gm[k][i] for k in range(2)),P(0))
        out.append(tr(r,N-1))
    for i in range(2):
        for j in range(i+1,2):
            out.append(tr(sum((g[i][k]*Ph[k][j]-g[j][k]*Ph[k][i] for k in range(2)),P(0)),N))
    for i in range(2):
        for j in range(2):
            for k in range(j+1,2):
                out.append(tr(d(g[i][j],W[k])-d(g[i][k],W[j]),N-1))
    return out
vs=sp.symbols('v1:3')
# g0 independently: Hessian of -sqrt(1-v^2) via sympy, then series
L0=-sp.sqrt(1-sum((v0[i]+eps*W[i])**2 for i in range(2)))
g0=[[None]*2 for _ in range(2)]
for i in range(2):
    for j in range(2):
        h=sp.diff(-sp.sqrt(1-sum(q**2 for q in vs)),vs[i],vs[j])
        hs=sp.series(h.subs({vs[k]:v0[k]+eps*W[k] for k in range(2)}),eps,0,N+1).removeO().subs(eps,1)
        g0[i][j]=tr(P(hs),N)
print("g0 residual nonzero terms:",sum(len(r.as_dict()) if not r.is_zero else 0 for r in resid(g0)))
mons=sorted(set(m for m in sp.itermonomials(G,N)), key=sp.default_sort_key)
cols=[(pi,m) for pi in range(len(pairs)) for m in mons]
rows={}; ent=[]
for ci,(pi,m) in enumerate(cols):
    i,j=pairs[pi]; Z=P(0)
    g=[[Z]*2 for _ in range(2)]; g[i][j]=P(m); g[j][i]=P(m)
    for ri,r in enumerate(resid(g)):
        for mm,c in r.as_dict().items():
            k=(ri,mm); rows.setdefault(k,len(rows)); ent.append((rows[k],ci,c))
nr,nc=len(rows),len(cols)
def M(colsel=None, extra=None):
    idx={c:i for i,c in enumerate(colsel)} if colsel else None
    ncc=len(colsel) if colsel else nc
    m=[[fmpq(0)]*ncc for _ in range(nr+(len(extra) if extra else 0))]
    for r,c,v in ent:
        if idx is None: m[r][c]=fmpq(int(v.p),int(v.q))
        elif c in idx: m[r][idx[c]]=fmpq(int(v.p),int(v.q))
    if extra:
        for e,c in enumerate(extra): m[nr+e][c]=fmpq(1)
    return fmpq_mat(len(m),ncc,[x for row in m for x in row])
deg=lambda m: sum(sp.Poly(m,*G).monoms()[0])
rA=M().rank()
wsel=[ci for ci,(pi,m) in enumerate(cols) if deg(m)==N]
rB=M(wsel).rank()
for jm in (1,N-1):
    sel=[ci for ci,(pi,m) in enumerate(cols) if deg(m)<=jm]
    print(f"[K2-2D {mode} N={N}] rankA={rA} #w={len(wsel)} rankB={rB} nullityB={len(wsel)-rB}  image dim in <= {jm}-jets = {M(extra=sel).rank()-rA}")
