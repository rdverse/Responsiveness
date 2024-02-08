class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


class globals():
    #Collect all images

    def __init__(self):
        self.PATH = 'chrisPP'
        self.newPATH = 'classChrisPP'
        self.extractor = self.initExtractor()

    def initExtractor(self):
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=
            "/home/redev/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
            device='cuda')
        return extractor

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):

        inputs, labels = (torch.stack(data['train']['features']).to(device),
                          data['train']['labels'])
        print(type(data['train']['features'][0]))
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    outputs = net(data['test']['features'])
    print(data['test']['files'])
    print(outputs)
