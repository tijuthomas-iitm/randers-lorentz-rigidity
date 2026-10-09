import sys, random
from fractions import Fraction as Fr
import helm_exact as H
from math import comb
from flint import fmpq_mat, fmpq
# ---------- K1: Theorem-3 multipliers must pass the exact truncated system ----------
def geom_inv(c,lin,D,nv):   # 1/(c + lin) as series, lin a poly with zero constant term
    one={tuple([0]*nv):Fr(1)}; r={}; term=dict(one)
    for k in range(D+1):
        r=H.add(r,term,Fr(1)/c*(Fr(-1)/c)**k); term=H.mul(term,lin,D)
    return r
def k1(mode,LH_builder,label,N=4):
    S=H.setup(3,N,1,mode); n,nv,D=3,S['nv'],S['D']
    w=lambda a: {tuple(1 if q==1+n+a else 0 for q in range(nv)):Fr(1)}
    v0=[S['v'][a][tuple([0]*nv)] for a in range(3)]
    LH=LH_builder(S,w,v0,nv,D)
    g=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            p=H.mul(S['gam3'],H.mul(S['v'][i],S['v'][j],D),D)
            if i==j: p=H.add(p,S['gam'])
            hess=H.diff(H.diff(LH,1+n+i),1+n+j)
            g[i][j]=H.deg(H.add(p,hess,Fr(1,7)),N)
    bad=sum(len(r) for _,r in H.residuals(S,g))
    print(f"K1 [{label}] nonzero residual terms of g0 + Hess(L_H)/7 : {bad}")
# uniform B_z: kernel span(e_t,e_z) -> L_H = k(v_z) = v_z^3 + v_z^4
k1('B', lambda S,w,v0,nv,D: H.add(H.mul(H.mul(S['v'][2],S['v'][2],D),S['v'][2],D), H.mul(H.mul(S['v'][2],S['v'][2],D),H.mul(S['v'][2],S['v'][2],D),D)), "uniform B, L_H=v_z^3+v_z^4")
# control: k(v_x) must FAIL
k1('B', lambda S,w,v0,nv,D: H.mul(H.mul(S['v'][0],S['v'][0],D),S['v'][0],D), "CONTROL uniform B, L_H=v_x^3 (should fail)")
# constant-kernel E along x1, E=E(t,x1): kernel span(e_y,e_z) -> L_H = v_y^2 / v_z
def LH_E(S,w,v0,nv,D):
    inv=geom_inv(v0[2],w(2),D,nv); return H.mul(H.mul(S['v'][1],S['v'][1],D),inv,D)
k1('kerconstE', LH_E, "non-uniform E(t,x) along x, L_H=v_y^2/v_z")
def LH_Ebad(S,w,v0,nv,D):
    inv=geom_inv(v0[2],w(2),D,nv); return H.mul(H.mul(S['v'][0],S['v'][0],D),inv,D)
k1('kerconstE', LH_Ebad, "CONTROL E along x, L_H=v_x^2/v_z (should fail)")
# linear plane wave A2=a(t-x1): kernel span((1,1,0,0),e_z): pairings 1-v1 and v3 -> L_H = v3^2/(1-v1)
def LH_pw(S,w,v0,nv,D):
    inv=geom_inv(1-v0[0],H.scal(w(0),-1),D,nv); return H.mul(H.mul(S['v'][2],S['v'][2],D),inv,D)
k1('pwlin', LH_pw, "linear plane wave, L_H=v_z^2/(1-v_x)")
k1('pwcirc', LH_pw, "CONTROL circular plane wave, same L_H (should fail)")
# ---------- K3: abstract w-block nullity for given (v0,f0,Phi0) ----------
def wblock_nullity(n,N,v0,f0,Phi0):
    nv=1+2*n; ms=[m for m in H.monos(nv,N) if sum(m)==N]
    pairs=[(i,j) for i in range(n) for j in range(i,n)]
    cols=[(p,m) for p in range(len(pairs)) for m in ms]; keys={}; ent=[]
    for ci,(p,m) in enumerate(cols):
        i,j=pairs[p]; g=[[{} for _ in range(n)] for _ in range(n)]; g[i][j]={m:Fr(1)}; g[j][i]={m:Fr(1)}
        R=[]
        for a,b in pairs:   # X g
            r=H.diff(g[a][b],0)
            for q in range(n): r=H.add(r,H.scal(H.diff(g[a][b],1+q),v0[q])); r=H.add(r,H.scal(H.diff(g[a][b],1+n+q),f0[q]))
            R.append(r)
        for a in range(n):
            for b in range(a+1,n):
                r={}
                for k in range(n): r=H.add(r,H.scal(g[a][k],Phi0[k][b])); r=H.add(r,H.scal(g[b][k],Phi0[k][a]),-1)
                R.append(r)
        for a in range(n):
            for b in range(n):
                for k in range(b+1,n): R.append(H.add(H.diff(g[a][b],1+n+k),H.diff(g[a][k],1+n+b),-1))
        for ri,r in enumerate(R):
            for mm,val in r.items():
                kk=(ri,mm); keys.setdefault(kk,len(keys)); ent.append((keys[kk],ci,val))
    M=[[fmpq(0)]*len(cols) for _ in range(len(keys))]
    for r,c,val in ent: M[r][c]=fmpq(val.numerator,val.denominator)
    return len(cols)-fmpq_mat(len(keys),len(cols),[x for row in M for x in row]).rank()
rnd=random.Random(5); R_=lambda: Fr(rnd.randint(-9,9),rnd.randint(1,5))
n,N=3,3
tests={
 "random real Phi0":[[R_() for _ in range(3)] for _ in range(3)],
 "complex pair eigen (rotation+scale)":[[Fr(1),Fr(-2),Fr(0)],[Fr(2),Fr(1),Fr(0)],[Fr(0),Fr(0),Fr(3)]],
 "repeated eigenvalue, nonderogatory (Jordan block)":[[Fr(2),Fr(1),Fr(0)],[Fr(0),Fr(2),Fr(0)],[Fr(0),Fr(0),Fr(5)]],
 "repeated eigenvalue, derogatory diag(2,2,5)":[[Fr(2),0,0],[0,Fr(2),0],[0,0,Fr(5)]],
 "scalar Phi0":[[Fr(3),0,0],[0,Fr(3),0],[0,0,Fr(3)]],
}
v0=[R_() for _ in range(3)]; f0=[R_() for _ in range(3)]
for k,Ph in tests.items():
    print(f"K3 [{k}] nullity={wblock_nullity(n,N,v0,f0,Ph)}  formula n*C(N+n,n)={n*comb(N+n,n)}")
