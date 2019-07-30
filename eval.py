import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def loss_on_loader(model, loader, M=1, device=device):
    total = 0
    total_stats = None
    batches = 0

    with torch.no_grad():
        for x, _ in loader:
            batches += 1
            x = x.to(device)
            loss, stats = model.elbo_loss(x, M=M, lbd=0.0)

            total += loss
            if total_stats is None:
                total_stats = stats
            else:
                for key in stats:
                    total_stats[key] += stats[key]

    total = total / batches
    for key in total_stats:
        total_stats[key] = total_stats[key] / batches
        
    return total, total_stats


# def eval_supervised(model, loader):
#     model.eval()
#     N = 0
#     score = 0
#     for x, y in loader:
#         x = x.to(device)
#         y = y.to(device)
#
#         pred = model.classify(x)
#         score += torch.sum(pred == y)
#         N += x.shape[0]
#     return float(score.item()) / float(N)
#
#
# def eval_unsupervised(model, loader):
#     model.eval()
#     probs, truth = [], []
#     for x, y in loader:
#         x = x.to(device)
#         truth.append(y)
#         probs.append(model.encode_y(x))
#     probs = torch.cat(probs, dim=0)
#     truth = torch.cat(truth, dim=0).to(device)
#
#     cluster = torch.argmax(probs, dim=1)
#     best = torch.argmax(probs, dim=0)
#     N = cluster.shape[0]
#     K = best.shape[0]
#     labels = torch.zeros(N, device=device, dtype=torch.int64)
#     for i in range(K):
#         labels[cluster == i] = truth[best[i]]
#
#     score = torch.sum(labels == truth)
#     return float(score.item()) / float(N)
#
#
# if __name__ == '__main__':
#     test_dataset = MNIST(train=False)
#     loader = DataLoader(test_dataset, batch_size=32, num_workers=2, drop_last=True)
#     model = MNISTGMVAE.load_from_drive(MNISTGMVAE)
#     print(eval_unsupervised(model, test_dataset))
