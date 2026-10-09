# Exact check of the structural genericity argument:
#  y = jets <= N-1, w = jets of order N.  B = A[:, w].  Claim: nullity(B) = n*C(N+n,n) when Phi0 has distinct eigenvalues,
#  and dim pi_{N-1}(ker A) = rank[A;pi]-rank A = 1.
import sys, sympy as sp
from math import comb
from fractions import Fraction as Fr
from flint import fmpq_mat, fmpq
import helm_exact as H
def build(n,N,seed,mode):
    S=H.setup(n,N,seed,mode); nv=S['nv']
    pairs=[(i,j) for i in range(n) for j in range(i,n)]
    ms=H.monos(nv,N); cols=[(pi,m) for pi in range(len(pairs)) for m in ms]
    keymap={}; ent=[]
    for ci,(pi,m) in enumerate(cols):
        i,j=pairs[pi]; g=[[{} for _ in range(n)] for _ in range(n)]
        g[i][j]={m:Fr(1)}; g[j][i]={m:Fr(1)}
        for ridx,(tag,r) in enumerate(H.residuals(S,g)):
            for mm,val in r.items():
                k=(ridx,mm); keymap.setdefault(k,len(keymap)); ent.append((keymap[k],ci,val))
    return S,cols,keymap,ent
def mat(nr,nc,ent,colsel=None):
    idx={c:i for i,c in enumerate(colsel)} if colsel is not None else None
    M=[[0]*(len(colsel) if colsel else nc) for _ in range(nr)]
    for r,c,v in ent:
        if idx is None: M[r][c]=v
        elif c in idx: M[r][idx[c]]=v
    rows=len(M); cc=len(M[0])
    return fmpq_mat(rows,cc,[fmpq(x.numerator,x.denominator) if isinstance(x,Fr) else fmpq(x) for row in M for x in row])
def phi0(S):
    n=S['n']; Z=tuple([0]*S['nv'])
    return sp.Matrix(n,n,lambda i,j: sp.Rational(S['Phi'][i][j].get(Z,0).numerator, S['Phi'][i][j].get(Z,0).denominator))
if __name__=="__main__":
    n,N,mode,seed=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
    S,cols,keymap,ent=build(n,N,seed,mode); nr,nc=len(keymap),len(cols)
    A=mat(nr,nc,ent); rA=A.rank()
    wcols=[ci for ci,(pi,m) in enumerate(cols) if sum(m)==N]
    B=mat(nr,nc,ent,wcols); rB=B.rank()
    P=phi0(S); lam=sp.symbols('l'); cp=P.charpoly(lam).as_expr()
    disc=sp.discriminant(cp,lam)
    ny=nc-len(wcols)
    print(f"[{mode} n={n} N={N} seed={seed}] rankA={rA} #w={len(wcols)} rankB={rB} nullityB={len(wcols)-rB} formula n*C(N+n,n)={n*comb(N+n,n)}  dim pi_(N-1)(K)= dim y - (rankA-rankB) = {ny-(rA-rB)}  disc(charpoly Phi0)={disc}  Phi0 eigen={sp.factor(cp)}")
