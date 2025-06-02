import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import EyeCnn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from DataResize import FaceDataLoader




def train(path,epo):
  transform=transforms.ToTensor()

  Dataload=FaceDataLoader(path,transform=transform)
  dataloader = DataLoader(Dataload, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model=EyeCnn().to(device)
  criterion = nn.MSELoss()
  optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

  diag = (224**2 + 224**2) ** 0.5  #Normalize the value

  for epochs in range(epo):
    model.train()
    running_loss=0.0
    total_error = 0.0
    total_samples = 0
    for image,label in dataloader:
      image=image.to(device)
      label=label.float().to(device)

      optimizer.zero_grad()
      output=model(image)
      loss=criterion(output,label)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()

       # Calculate Euclidean error per landmark point (batch_size, 68, 2)
      # The output of the model is (batch_size, 136) and the label is (batch_size, 136)
      batch_size = label.size(0)
      error = torch.sqrt(((output - label).view(batch_size, -1, 2) ** 2).sum(dim=2))
      mean_error_per_sample = error.mean(dim=1)
      batch_avg_error = mean_error_per_sample.mean().item()

      total_error += batch_avg_error * batch_size
      total_samples += batch_size


    epoch_loss = running_loss / len(Dataload)
    epoch_avg_error = total_error / total_samples
    accuracy = (1 - epoch_avg_error / diag) * 100

    print(f"Epoch {epochs + 1}/{epo}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

   # print(f"Epoch {epochs+1}/{epo}, Loss: {running_loss/len(Dataload):.4f}")

  torch.save(model.state_dict(),"/content/drive/MyDrive/weights.pth")