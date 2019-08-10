import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets import JointDataset
from transforms import SetRotation
from vae import PCVAE
from eval import loss_on_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_unsupervised(model, optimizer, scheduler, train_loader, test_loader,
                       num_epochs=1000, M=1, lbd=0.0):
    print('Training your model!\n')
    model.train()

    best_params = None
    best_loss = float('inf')
    logs = defaultdict(list)

    try:
        for epoch in range(num_epochs):
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()

                loss, stats = model.elbo_loss(x, M=M, lbd=lbd)

                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss, train_stats = loss_on_loader(model, train_loader, M=M, device=device)
            test_loss, test_stats = loss_on_loader(model, test_loader, M=M, device=device)
            logs['train_loss'].append(train_loss)
            logs['test_loss'].append(test_loss)
            for key, val in train_stats.items():
                logs['train'+key].append(val)
            for key, val in test_stats.items():
                logs['test'+key].append(val)

            if test_loss < best_loss:
                best_loss = test_loss
                best_params = model.state_dict()

            if (epoch % 25) == 1 or epoch == num_epochs-1:
                print("Epoch {epoch}\ntrain loss={train_loss}, train stats={train_stats}\n"
                      "test loss={test_loss}, test stats={test_stats}"
                       .format(epoch=epoch, train_loss=train_loss, train_stats=train_stats,
                               test_loss=test_loss, test_stats=test_stats)
                     )

    except KeyboardInterrupt:
        pass

    model.load_state_dict(best_params)
    model.eval()
    model.cpu()

    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')

    print('Saving training logs...', end='')
    np_logs = np.stack([np.array(item) for item in logs.values()], axis=0)
    np.save('train_logs', np_logs)
    print('done.')


def train_vae(model, train_dataset, test_dataset, M=1, lbd=0.0, num_epochs=1000):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=True)

    optimizer = Adam(model.parameters(), lr=2e-4)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

    train_unsupervised(model, optimizer, scheduler, train_loader, test_loader,
                       lbd=lbd, M=M, num_epochs=num_epochs)


if __name__ == '__main__':
    train_dataset = JointDataset(filter=1, transform_shapenet=SetRotation((0, math.acos(0), 0)))
    test_dataset = JointDataset(filter=1, test=True, transform_shapenet=SetRotation((0, math.acos(0), 0)))
    model = PCVAE(decoder=[512, 1024, 1024, 2048])
    model.to(device)
    train_vae(model, train_dataset, test_dataset, num_epochs=3000, M=1)


# def train_m2(model, train_dataset, drop_labels=0.0, log_every=200):
#     N = len(train_dataset)
#     t = int(torch.clamp(torch.tensor(N * drop_labels), 1, N-1).item())
#     unlabeled_dataset, labeled_dataset = random_split(train_dataset, (t, N-t))
#
#     unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
#     labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
#
#     optimizer = Adam(model.parameters(), lr=1e-4)
#     train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
#                         p=drop_labels, lbd=0.0, num_epochs=2000, log_every=log_every)

# def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
#                          num_epochs=1000, p=0.0, M=1, lbd=0.0, log_every=200):
#     print('Training your model!\n')
#     model.train()
#
#     global_step = 0
#     best_params = None
#     best_loss = INF
#
#     try:
#         for epoch in range(num_epochs):
#             labeled_iter = iter(labeled_loader)
#             unlabeled_iter = iter(unlabeled_loader)
#             while True:
#                 global_step += 1
#                 optimizer.zero_grad()
#
#                 drop_labels = torch.bernoulli(torch.tensor(p)).item()
#                 draw_from = unlabeled_iter if drop_labels else labeled_iter
#
#                 try:
#                     x, y = next(draw_from)
#                     x = x.to(device)
#                     y = y.to(device)
#                 except StopIteration:
#                     break
#
#                 if drop_labels:
#                     loss, stats = model.elbo_unknown_y(x, M=M, lbd=lbd)
#                 else:
#                     loss, stats = model.elbo_known_y(x, y, M=M, lbd=lbd)
#                 loss.backward()
#                 optimizer.step()
#
#                 if (global_step%log_every) == 1:
#                     print("global step: %d (epoch: %d), loss: %f %s" %
#                           (global_step, epoch, loss.item(), stats))
#
#     except KeyboardInterrupt:
#         pass
#
#     model.eval()
#     print('Saving model to drive...', end='')
#     model.cpu()
#     model.save_to_drive()
#     print('done.')

# def train_unsupervised_with_eval(model, optimizer, train_loader,
#                        test_loader=None, num_epochs=1000, M=1, lbd=0.0, log_every=200):
#     print('Training your model!\n')
#     model.train()
#
#     global_step = 0
#     best_params = None
#     best_loss = INF
#
#     try:
#         for epoch in range(num_epochs):
#             for inum, (x, _) in enumerate(train_loader):
#                 x = x.to(device)
#
#                 global_step += 1
#                 optimizer.zero_grad()
#
#                 loss, stats = model.elbo_loss(x, M=M, lbd=lbd)
#
#                 loss.backward()
#                 optimizer.step()
#
#                 if (global_step%log_every) == 1:
#                     print("global step: %d (epoch: %d, step: %d), loss: %f %s" %
#                           (global_step, epoch, inum, loss.item(), stats))
#
#             if test_loader:
#                 model.eval()
#                 score = eval_unsupervised(model, test_loader)
#                 print("Epoch {}, acc: {}".format(epoch, score))
#                 model.train()
#
#     except KeyboardInterrupt:
#         pass
#
#     model.eval()
#     print('Saving model to drive...', end='')
#     model.cpu()
#     model.save_to_drive()
#     print('done.')
