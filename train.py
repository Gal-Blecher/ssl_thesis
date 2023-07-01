import torch
import torch.optim as optim
from config import setup

def train_vae(model, data, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data['train_loader']
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=setup['lr'])

    for epoch in range(setup['n_epochs']):
        recon_losses = []
        kl_losses = []
        total_loss = 0

        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            model(images)

            loss = model.recon_loss + 0.000000001 * model.kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_losses.append(model.recon_loss.item())
            kl_losses.append(model.kl_loss.item())
            total_loss += loss.item()

            # Print training progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{setup['n_epochs']}]"
                      f"Reconstruction Loss: {recon_losses[-1]:.4f}, KL Divergence: {kl_losses[-1]:.4f}")

        # Print epoch summary
        avg_recon_loss = sum(recon_losses) / len(recon_losses)
        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        print(f"\nEpoch [{epoch + 1}/{setup['n_epochs']}] Summary:")
        print(f"  Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  Average KL Divergence Loss: {avg_kl_loss:.4f}")

        # Save the model
        torch.save(model, f'{save_path}/model.pkl')

    print("\nTraining completed!")
