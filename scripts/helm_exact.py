import sys, time
from fractions import Fraction as Fr
from math import isqrt
import random
from flint import fmpq_mat, fmpq
def monos(nv,D):
    out=[]
    def rec(i,left,cur):
        if i==nv: out.append(tuple(cur)); return
        for e in range(left+1): rec(i+1,left-e,cur+[e])
    rec(0,D,[]); return out
def add(a,b,c=1):
    c=Fr(c); r=dict(a)
    for k,v in b.items():
        nv_=r.get(k,0)+c*v
        if nv_==0: r.pop(k,None)
        else: r[k]=nv_
    return r
def mul(a,b,D):
    r={}
    for k1,v1 in a.items():
        d1=sum(k1)
        for k2,v2 in b.items():
            if d1+sum(k2)>D: continue
            k=tuple(x+y for x,y in zip(k1,k2)); r[k]=r.get(k,0)+v1*v2
    return {k:v for k,v in r.items() if v!=0}
def scal(a,c): c=Fr(c); return {k:v*c for k,v in a.items()}
def diff(a,i):
    r={}
    for k,v in a.items():
        if k[i]>0:
            kk=list(k); kk[i]-=1; kk=tuple(kk); r[kk]=r.get(kk,0)+v*k[i]
    return r
def deg(a,D): return {k:v for k,v in a.items() if sum(k)<=D}
def setup(n,N,seed,mode):
    nv=1+2*n; D=N+3; rnd=random.Random(seed)
    Z=tuple([0]*nv); one={Z:Fr(1)}
    def var(i):
        k=[0]*nv; k[i]=1; return {tuple(k):Fr(1)}
    t=var(0); xs=[var(1+a) for a in range(n)]; ws=[var(1+n+a) for a in range(n)]
    VL={3:[([Fr(1,2)]*3,Fr(1,2)),([Fr(1,5),Fr(2,5),Fr(2,5)],Fr(4,5)),([Fr(2,9),Fr(4,9),Fr(5,9)],Fr(2,3))],
        2:[([Fr(2,3),Fr(1,3)],Fr(2,3)),([Fr(4,9),Fr(4,9)],Fr(7,9)),([Fr(1,3),Fr(2,3)],Fr(2,3))]}
    v0,r0=VL[n][seed%3]
    s0=r0*r0
    v=[add({Z:v0[a]},ws[a]) for a in range(n)]
    v2={}
    for a in range(n): v2=add(v2,mul(v[a],v[a],D))
    s=add(one,v2,-1)
    u=scal(add(s,{Z:-s0}),1/s0)
    def powser(p,pf):   # (s0(1+u))^p, p half-integer; pf = s0^p exactly
        res={}; term=dict(one); coef=Fr(1)
        for m in range(D+1):
            res=add(res,term,coef)
            coef=coef*(p-m)/(m+1)
            term=mul(term,u,D)
        return scal(res,pf)
    sq=powser(Fr(1,2),r0); gam=powser(Fr(-1,2),1/r0); gam3=powser(Fr(-3,2),1/r0**3)
    def pot(d_):
        r={}
        for k in monos(1+n,d_):
            kk=tuple(list(k)+[0]*n); c=rnd.randint(-3,3)
            if c: r[kk]=Fr(c)
        return r
    def lin(coefs):
        r={}
        for k,c in coefs.items():
            kk=[0]*nv; kk[k]=1; r[tuple(kk)]=Fr(c)
        return r
    if mode=='gen':
        Phi_=pot(3); A_=[pot(3) for a in range(n)]
    elif mode=='B':   # uniform B along x3
        Phi_={}; A_=[{} for _ in range(n)]; A_[0]=lin({2:Fr(-3,5)}); A_[1]=lin({1:Fr(3,5)})
    elif mode=='EB':
        Phi_=lin({3:Fr(-4,5)}); A_=[{} for _ in range(n)]; A_[0]=lin({2:Fr(-3,5)}); A_[1]=lin({1:Fr(3,5)})
    elif mode in ('pwlin','pwcirc'):   # u = t - x1 ; A2 = a(u), A3 = b(u)
        u=add(lin({0:1}),lin({1:-1})); u2=mul(u,u,9); u3=mul(u2,u,9)
        Phi_={}; A_=[{} for _ in range(n)]
        A_[1]=add(add(scal(u,Fr(7,10)),scal(u2,Fr(2,5))),scal(u3,Fr(3,10)))
        if mode=='pwcirc': A_[2]=add(add(scal(u,Fr(-1,2)),scal(u2,Fr(3,5))),scal(u3,Fr(-1,5)))
    elif mode=='kerconst':  # F = b(x1,x2) dx1^dx2 with nonuniform b: A2 = cubic in x1,x2 only
        Phi_={}; A_=[{} for _ in range(n)]
        r={}
        for k in monos(1+n,3):
            if k[0]==0 and k[3]==0:
                c=rnd.randint(-3,3)
                if c: r[tuple(list(k)+[0]*n)]=Fr(c)
        A_[1]=r
    elif mode=='kerconstE':  # F = b(t,x1) dt^dx1 : E along x1 depending on (t,x1); Phi = -int b
        r={}
        for k in monos(1+n,3):
            if k[2]==0 and k[3]==0:
                c=rnd.randint(-3,3)
                if c: r[tuple(list(k)+[0]*n)]=Fr(c)
        Phi_=r; A_=[{} for _ in range(n)]
    elif mode=='radial':   # E = -grad(c|x|^2): radial linear field
        Phi_={tuple([0,2,0,0]+[0]*n):Fr(2,3),tuple([0,0,2,0]+[0]*n):Fr(2,3),tuple([0,0,0,2]+[0]*n):Fr(2,3)}; A_=[{} for _ in range(n)]
    elif mode=='radialshift':  # Phi = (2/3)|x - xc|^2 , xc=(1/3,-1/3,2/3)
        xc=[Fr(1,3),Fr(-1,3),Fr(2,3)]; Phi_={}
        for a in range(3):
            q=add(lin({1+a:1}),{tuple([0]*nv):-xc[a]}); Phi_=add(Phi_,mul(q,q,9),Fr(2,3))
        A_=[{} for _ in range(n)]
    elif mode=='coulombish':   # 1/r-like: Phi = 1/|x-xc| Taylor-truncated? use Phi = (|x-xc|^2)^2 quartic radial
        xc=[Fr(1,3),Fr(-1,3),Fr(2,3)]; r2={}
        for a in range(3):
            q=add(lin({1+a:1}),{tuple([0]*nv):-xc[a]}); r2=add(r2,mul(q,q,9))
        Phi_=scal(mul(r2,r2,9),Fr(1,5)); A_=[{} for _ in range(n)]
    elif mode=='axial':   # axisymmetric static E: Phi = phi(rho^2, z) about axis through xc along z; Killing: d_t, rotation about axis -> predicted 2 -> non-rigid?
        xc=[Fr(1,3),Fr(-1,3)]; rho2={}
        for a in range(2):
            q=add(lin({1+a:1}),{tuple([0]*nv):-xc[a]}); rho2=add(rho2,mul(q,q,9))
        z=add(lin({3:1}),{tuple([0]*nv):Fr(-2,3)})
        Phi_=add(scal(rho2,Fr(2,3)),add(scal(mul(rho2,z,9),Fr(1,2)),scal(mul(z,z,9),Fr(-1,3)))); A_=[{} for _ in range(n)]
    elif mode=='axialB':  # axisymmetric magnetic, A_phi only: A = a(rho^2,z)(-(y-yc),(x-xc),0); Killing in kernel: d_t, rotation? (iota_rot F = -d(rho*A_phi) != 0 in general)
        xc=[Fr(1,3),Fr(-1,3)]; X0=add(lin({1:1}),{tuple([0]*nv):-xc[0]}); Y0=add(lin({2:1}),{tuple([0]*nv):-xc[1]})
        rho2=add(mul(X0,X0,9),mul(Y0,Y0,9)); z=add(lin({3:1}),{tuple([0]*nv):Fr(-2,3)})
        a=add({tuple([0]*nv):Fr(1,2)},add(scal(z,Fr(1,3)),scal(rho2,Fr(-1,4))))
        A_=[scal(mul(a,Y0,9),-1),mul(a,X0,9),{}]; Phi_={}
    elif mode=='cyl':   # Phi = phi(rho) about axis through (1/3,-1/3) along z, z-independent: kernel Killing = rotation + d_z -> predicted NON-rigid
        xc=[Fr(1,3),Fr(-1,3)]; rho2={}
        for a in range(2):
            q=add(lin({1+a:1}),{tuple([0]*nv):-xc[a]}); rho2=add(rho2,mul(q,q,9))
        Phi_=add(scal(rho2,Fr(2,3)),scal(mul(rho2,rho2,9),Fr(-1,4))); A_=[{} for _ in range(n)]
    elif mode=='coulomb':   # Phi = 1/|x - xc|, |xc|=1, Taylor to degree D+1 (exact rationals)
        import sympy as _sp
        xs_=_sp.symbols('X1:4'); e_=_sp.Symbol('e'); xc=[_sp.Rational(1,3),_sp.Rational(2,3),_sp.Rational(2,3)]
        ser=_sp.series(1/_sp.sqrt(sum((e_*xs_[i]-xc[i])**2 for i in range(3))),e_,0,D+2).removeO().subs(e_,1)
        Phi_={}
        for mon,c in _sp.Poly(_sp.expand(ser),*xs_).as_dict().items():
            Phi_[tuple([0]+list(mon)+[0]*n)]=Fr(int(_sp.Rational(c).p),int(_sp.Rational(c).q))
        A_=[{} for _ in range(n)]
    elif mode in ('radial2','coulomb2'):   # 2+1 central fields, centre xc at unit distance
        import sympy as _sp
        X_=_sp.symbols('X1:3'); e_=_sp.Symbol('e'); xc=[_sp.Rational(3,5),_sp.Rational(4,5)]
        r2=sum((e_*X_[i]-xc[i])**2 for i in range(2))
        expr=_sp.Rational(2,3)*r2+_sp.Rational(-1,5)*r2**2 if mode=='radial2' else 1/_sp.sqrt(r2)
        ser=_sp.series(expr,e_,0,D+2).removeO().subs(e_,1)
        Phi_={}
        for mon,c in _sp.Poly(_sp.expand(ser),*X_).as_dict().items():
            Phi_[tuple([0]+list(mon)+[0]*n)]=Fr(int(_sp.Rational(c).p),int(_sp.Rational(c).q))
        A_=[{} for _ in range(n)]
    elif mode=='helical':  # Beltrami-type rotating B: A=(1-z^2/2, z-z^3/6, 0)
        Phi_={}; A_=[{} for _ in range(n)]
        A_[0]={tuple([0,0,0,0]+[0]*n):Fr(1),tuple([0,0,0,2]+[0]*n):Fr(-1,2)}
        A_[1]={tuple([0,0,0,1]+[0]*n):Fr(1),tuple([0,0,0,3]+[0]*n):Fr(-1,6)}
    elif mode in ('BzEz','Bperp'):
        r={}
        for k in monos(1+n,3):
            if k[0]==0 and k[3]==0:
                c=rnd.randint(-3,3)
                if c: r[tuple(list(k)+[0]*n)]=Fr(c)
        A_=[{} for _ in range(n)]
        if mode=='BzEz': A_[1]=r; Phi_=lin({3:Fr(-4,5)})       # B_z(x,y) nonuniform + uniform E_z
        else: A_[2]=r; Phi_={}                                   # B=(dA_z/dy,-dA_z/dx,0): rotating in-plane, z- and t-symmetric
    elif mode=='static':
        Phi_={}; A_=[]
        for a in range(n):
            r={}
            for k in monos(1+n,3):
                if k[0]==0:
                    c=rnd.randint(-3,3)
                    if c: r[tuple(list(k)+[0]*n)]=Fr(c)
            A_.append(r)
    E=[add(scal(diff(Phi_,1+a),-1),diff(A_[a],0),-1) for a in range(n)]
    if n==3: B=[add(diff(A_[(i+2)%3],1+(i+1)%3),diff(A_[(i+1)%3],1+(i+2)%3),-1) for i in range(3)]
    else: B=[add(diff(A_[1],1),diff(A_[0],2),-1)]
    def cross(i):
        if n==3:
            a,b=(i+1)%3,(i+2)%3
            return add(mul(v[a],B[b],D),mul(v[b],B[a],D),-1)
        return mul(v[1],B[0],D) if i==0 else scal(mul(v[0],B[0],D),-1)
    vE={}
    for a in range(n): vE=add(vE,mul(v[a],E[a],D))
    f=[mul(sq,add(add(E[i],cross(i)),mul(v[i],vE,D),-1),D) for i in range(n)]
    def Gam(p):
        r=diff(p,0)
        for a in range(n): r=add(r,mul(v[a],diff(p,1+a),D))
        for a in range(n): r=add(r,mul(f[a],diff(p,1+n+a),D))
        return r
    df=[[diff(f[k],1+n+j) for j in range(n)] for k in range(n)]
    Phi=[[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            r=scal(diff(f[i],1+j),-1)
            r=add(r,Gam(df[i][j]),Fr(1,2))
            for k in range(n): r=add(r,mul(df[i][k],df[k][j],D),Fr(-1,4))
            Phi[i][j]=r
    return dict(n=n,N=N,nv=nv,D=D,df=df,Phi=Phi,Gam=Gam,gam=gam,gam3=gam3,v=v)
def residuals(S,g):
    n,N,D,df,Phi,Gam=S['n'],S['N'],S['D'],S['df'],S['Phi'],S['Gam']
    pairs=[(i,j) for i in range(n) for j in range(i,n)]
    R=[]
    for (i,j) in pairs:
        r=Gam(g[i][j])
        for k in range(n):
            r=add(r,mul(g[i][k],df[k][j],D),Fr(1,2)); r=add(r,mul(g[j][k],df[k][i],D),Fr(1,2))
        R.append(('H2',deg(r,N-1)))
    for i in range(n):
        for j in range(i+1,n):
            r={}
            for k in range(n):
                r=add(r,mul(g[i][k],Phi[k][j],D)); r=add(r,mul(g[j][k],Phi[k][i],D),-1)
            R.append(('H3',deg(r,N)))
    for i in range(n):
        for j in range(n):
            for k in range(j+1,n):
                r=add(diff(g[i][j],1+n+k),diff(g[i][k],1+n+j),-1)
                R.append(('H4',deg(r,N-1)))
    return R
def certify(n,N,seed,mode,jetmax=1):
    t0=time.time()
    S=setup(n,N,seed,mode); nv=S['nv']; D=S['D']
    # validation: exact zero residual for g0
    g0=[[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p=mul(S['gam3'],mul(S['v'][i],S['v'][j],D),D)
            if i==j: p=add(p,S['gam'])
            g0[i][j]=deg(p,N)
    bad=sum(len(r) for _,r in residuals(S,g0))
    print(f"[{mode} n={n} N={N}] g0 exact residual terms: {bad}  (must be 0)")
    pairs=[(i,j) for i in range(n) for j in range(i,n)]
    ms=monos(nv,N); cols=[(pi,m) for pi in range(len(pairs)) for m in ms]
    keymap={}; entries=[]
    for ci,(pi,m) in enumerate(cols):
        i,j=pairs[pi]
        g=[[{} for _ in range(n)] for _ in range(n)]
        g[i][j]={m:Fr(1)}; g[j][i]={m:Fr(1)}
        for ridx,(tag,r) in enumerate(residuals(S,g)):
            for mm,val in r.items():
                key=(ridx,mm)
                if key not in keymap: keymap[key]=len(keymap)
                entries.append((keymap[key],ci,val))
    nr,nc=len(keymap),len(cols)
    rows=[[fmpq(0)]*nc for _ in range(nr)]
    for r,c,v_ in entries: rows[r][c]=fmpq(v_.numerator,v_.denominator)
    A=fmpq_mat(nr,nc,[x for row in rows for x in row])
    rA=A.rank()
    sel=[ci for ci,(pi,m) in enumerate(cols) if sum(m)<=jetmax]
    rows2=rows+[[fmpq(1) if c==ci else fmpq(0) for c in range(nc)] for ci in sel]
    A2=fmpq_mat(len(rows2),nc,[x for row in rows2 for x in row])
    rA2=A2.rank()
    print(f"[{mode} n={n} N={N}] rows={nr} cols={nc} rank A={rA}  rank[A;pi_{jetmax}]={rA2}  => dim image in {jetmax}-jets = {rA2-rA}  ({time.time()-t0:.0f}s)")
    return rA2-rA
if __name__=="__main__":
    n=int(sys.argv[1]); N=int(sys.argv[2]); mode=sys.argv[3]; seed=int(sys.argv[4]) if len(sys.argv)>4 else 1
    jm=int(sys.argv[5]) if len(sys.argv)>5 else 1
    certify(n,N,seed,mode,jm)
