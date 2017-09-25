import torch
import copy
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

X = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
n_particles = 50

def f(x, model):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    return loss.data[0]

lb = 0.
ub = 1.0
diff = abs(ub-lb)

pos = []
vel = []
x = []
for i in range(n_particles):
    pos += [[torch.rand(param.size()) for param in model.parameters()]]
    x += [copy.deepcopy(pos[i])]
    vel += [[(2*torch.rand(param.size())*diff - diff) 
            for param in model.parameters()]]

fg = 1e10
for i in range(n_particles):
    for j, param in enumerate(model.parameters()):
        param.data = pos[i][j]

    fp = f(X, model)

    if fp < fg:
        g = copy.deepcopy(pos[i])
        fg = fp

w, phig, phip = 0.9,0.3,0.5


# https://en.wikipedia.org/wiki/Particle_swarm_optimization

for t in range(5000):
    for i in range(n_particles):
        for j, param in enumerate(model.parameters()):
            param.data = pos[i][j]
        fp = f(X, model) 

        for j, param in enumerate(model.parameters()):
            rp = torch.rand(param.size())
            rg = torch.rand(param.size())

            vel[i][j] = (w * vel[i][j] + phip*rp*(pos[i][j] - x[i][j]) 
                + phig*rg*(g[j] - x[i][j])) 

            x[i][j] = x[i][j] + vel[i][j]

            param.data = x[i][j]

        fx = f(X, model)

        if fx < fp:
            pos[i] = copy.deepcopy(x[i])

            if fp < fg:
                g = copy.deepcopy(pos[i])
                fg = fp

    print "%d - loss: %.3f" % (t, fg)

