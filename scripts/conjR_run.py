from conjR import *
Z=lambda q:0.0
def run3(name,phi,A,rigid,n=3,K=4):
    x0=jnp.array([0.31,-0.22,0.41][:n]); v0=jnp.array([0.23,0.17,-0.12][:n])
    d=run(name,n,phi,A,(0.17,x0,v0),K); print("    known:",rigid,flush=True)
n=3
ph=randpoly(4,3); Ax=[randpoly(4,3) for _ in range(3)]
run3("generic random",ph,lambda q:jnp.array([a(q) for a in Ax]),"rigid")
run3("uniform B",Z,lambda q:jnp.array([-q[2]/2,q[1]/2,0.0]),"NON-rigid")
run3("uniform E",lambda q:-0.4*q[1]-0.3*q[3],lambda q:jnp.zeros(3),"NON-rigid")
run3("uniform E||B",lambda q:-0.4*q[3],lambda q:jnp.array([-q[2]/2,q[1]/2,0.0]),"rigid")
run3("crossed E⊥B",lambda q:-0.3*q[1],lambda q:jnp.array([-q[2]/2,q[1]/2,0.0]),"NON-rigid")
g=lambda u:u**2/2+u**3/5
run3("linear plane wave",Z,lambda q:jnp.array([g(q[0]-q[3]),0.0,0.0]),"NON-rigid")
run3("circular plane wave",Z,lambda q:jnp.array([jnp.cos(q[0]-q[3]),jnp.sin(q[0]-q[3]),0.0]),"rigid")
run3("E_x(x) nonuniform",lambda q:-(q[1]**2/2+q[1]**4/4),lambda q:jnp.zeros(3),"NON-rigid")
run3("B_z(x,y) static, fixed direction",Z,lambda q:jnp.array([0.0,q[1]+q[1]**3/3+q[1]**2*q[2]/2,0.0]),"NON-rigid")
pa=randpoly(4,3)
run3("static random magnetic",Z,lambda q:jnp.array([a(q.at[0].set(0.0)) for a in Ax]),"rigid")
run3("axisym electrostatic",lambda q:q[1]**2+q[2]**2+0.5*(q[1]**2+q[2]**2)**2+0.3*q[3]**2+0.2*q[3]*(q[1]**2+q[2]**2),lambda q:jnp.zeros(3),"rigid")
run3("axisym magnetic A_phi",Z,lambda q:(1+q[3]+q[1]**2+q[2]**2)*jnp.array([-q[2],q[1],0.0]),"rigid")
run3("helical Beltrami B",Z,lambda q:jnp.array([jnp.cos(q[3]),jnp.sin(q[3]),0.0]),"rigid")
run3("E_z uniform + B_z(x,y)",lambda q:-0.3*q[3],lambda q:jnp.array([0.0,q[1]+q[1]**3/3+q[1]**2*q[2]/2,0.0]),"rigid")
run3("cylindrical phi(rho)",lambda q:q[1]**2+q[2]**2+0.5*(q[1]**2+q[2]**2)**2,lambda q:jnp.zeros(3),"rigid")
xc=jnp.array([1.0,0.0,0.0])
rr2=lambda q:jnp.sum((q[1:]-xc)**2)
run3("central quadratic 3+1",lambda q:rr2(q),lambda q:jnp.zeros(3),"NON-rigid(jet)")
run3("central quartic 3+1",lambda q:rr2(q)+0.4*rr2(q)**2,lambda q:jnp.zeros(3),"NON-rigid(jet)")
run3("Coulomb 3+1",lambda q:1/jnp.sqrt(rr2(q)),lambda q:jnp.zeros(3),"NON-rigid(jet)")
# 2+1
xc2=jnp.array([0.6,0.8]); r2=lambda q:jnp.sum((q[1:]-xc2)**2)
ph2=randpoly(3,3); A2=[randpoly(3,3) for _ in range(2)]
run3("2+1 generic",ph2,lambda q:jnp.array([a(q) for a in A2]),"rigid",n=2)
run3("2+1 uniform B",Z,lambda q:jnp.array([-q[2]/2,q[1]/2]),"rigid",n=2)
run3("2+1 central quartic",lambda q:r2(q)+0.4*r2(q)**2,lambda q:jnp.zeros(2),"rigid",n=2)
run3("2+1 Coulomb",lambda q:1/jnp.sqrt(r2(q)),lambda q:jnp.zeros(2),"rigid",n=2)
run3("2+1 uniform E",lambda q:-0.4*q[1],lambda q:jnp.zeros(2),"?",n=2)
