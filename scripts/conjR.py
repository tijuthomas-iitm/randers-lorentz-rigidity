import jax, jax.numpy as jnp, numpy as np, itertools
jax.config.update("jax_enable_x64", True)
rng=np.random.default_rng(3)
def randpoly(n1,deg,scale=0.5):
    # random polynomial in n1 vars (t,x..), degree<=deg
    exps=[e for e in itertools.product(range(deg+1),repeat=n1) if sum(e)<=deg]
    c=jnp.array(rng.normal(size=len(exps))*scale); E=jnp.array(exps,dtype=float)
    return lambda q: jnp.sum(c*jnp.prod(q[None,:]**E,axis=1))
def make(n,phi,A):
    # phi(q), A(q)->n-vector, q=(t,x)
    def f(t,x,v):
        q=jnp.concatenate([jnp.array([t]),x])
        dphi=jax.grad(phi)(q); JA=jax.jacfwd(A)(q)   # JA[i,mu]=d_mu A_i
        E=-dphi[1:]-JA[:,0]
        Fm=JA[:,1:].T-JA[:,1:]   # Fm[i,j]=d_i A_j - d_j A_i
        return jnp.sqrt(1-v@v)*(E+Fm@v-v*(v@E))
    return f
def chain(f,n,K,pt):
    t0,x0,v0=pt
    Gc=lambda t,x,v: -0.5*jax.jacfwd(f,argnums=2)(t,x,v)
    def cov(M):
        def g(t,x,v):
            a=f(t,x,v)
            Mv,dM=jax.jvp(M,(t,x,v),(1.0,v,a))
            G=Gc(t,x,v)
            return dM+G@Mv-Mv@G
        return g
    def Gam(M):
        def g(t,x,v):
            a=f(t,x,v); return jax.jvp(M,(t,x,v),(1.0,v,a))[1]
        return g
    fx=lambda t,x,v: jax.jacfwd(f,argnums=1)(t,x,v)
    Phi=lambda t,x,v: -fx(t,x,v)-Gc(t,x,v)@Gc(t,x,v)-Gam(Gc)(t,x,v)
    Ms=[Phi]
    for k in range(K): Ms.append(cov(Ms[-1]))
    return [np.array(jax.jit(M)(t0,x0,v0)) for M in Ms],np.array(Gc(t0,x0,v0))
def algdim(Ms,tol=1e-8):
    n=Ms[0].shape[0]
    Ms=[M/np.linalg.norm(M) for M in Ms if np.linalg.norm(M)>1e-12]
    if not Ms: return 0
    def basis(L):
        A=np.array([m.ravel() for m in L]); u,s,vt=np.linalg.svd(A)
        r=int((s>tol*s[0]).sum()); return [vt[i].reshape(n,n) for i in range(r)]
    B=basis(Ms)
    for _ in range(6):
        new=basis(B+[a@b for a in B for b in Ms])
        if len(new)==len(B): break
        B=new
    return len(B)
def invsub(Ms,tol=1e-7):
    n=Ms[0].shape[0]; Ms=[M/np.linalg.norm(M) for M in Ms if np.linalg.norm(M)>1e-12]
    Mr=sum(rng.normal()*M for M in Ms); w,V=np.linalg.eig(Mr); found=[]
    for r in range(1,n):
        for S in itertools.combinations(range(n),r):
            U=V[:,S]; Q,_=np.linalg.qr(U)
            res=max(np.linalg.norm(Q@(Q.conj().T@(M@Q))-M@Q) for M in Ms)
            if res<tol: found.append(S)
    return found,np.min(np.abs(np.subtract.outer(w,w)+np.eye(n)*9))
def run(name,n,phi,A,pt,K=4):
    f=make(n,phi,A); Ms,_=chain(f,n,K,pt)
    d=algdim(Ms); sub,gap=invsub(Ms)
    print(f"{name:34s} n={n} algdim={d}/{n*n} invsub={len(sub)} {'REDUCIBLE' if d<n*n else 'irreducible'}",flush=True)
    return d
