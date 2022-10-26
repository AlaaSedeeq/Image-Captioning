#########################################
#            Trainer Class              #
#########################################

class Trainer:
    
    def __init__(self, model, optimizer, criterion, scheduler=None, load_path=None):
        self.__class__.__name__ = "PyTorch Trainer"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        tqdm.refresh
        ## Setup Metric class
        self.metrics = namedtuple('Metric', ['loss', 'train_error', 'val_error'])
        
        # if model exist
        if load_path:
            self.model = torch.load(load_path)
        
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((356, 356)),
            transforms.RandomCrop((350, 350)),
            transforms.CenterCrop(299),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5])
        ])
        
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def run(self, train_loader, n_epochs=100, save_model=True):
        
        min_valid_loss = np.inf
        ## Setup Metric class
        Metric = namedtuple('Metric', ['loss', 'train_error', 'val_error'])
        self.metrics = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train() 
        
        # Only finetune the CNN
        for name, param in self.model.Encoder._feature_extr.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = True

        if load_model:
            print('Loading checkpoint')
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']

        for epoch in range(n_epochs):
            if save_model:
                checkpoint = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                }
                print('Saving checkpoint')
                torch.save(state, filename)

            epoch_loss, correct = 0.0
            data_iter = enumerate(train_loader)
            t_prog_bar = tqdm(range(len(train_loader)))
#             lr = self.scheduler.get_last_lr()[0]
            lr = self.optimizer.param_groups[0]['lr']
            for step in t_prog_bar: # iter over batches
                    
                    outputs = model(imgs, captions[:-1])
                    loss = criterion(
                        outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
                    )

                    writer.add_scalar('Training loss', loss.item(), global_step=step)
                    step += 1

                    optimizer.zero_grad()
                    loss.backward(loss)
                    optimizer.step()
                
                batch_idx, (data, captions) = next(data_iter) # get the input images and their corresponding captions
                self.optimizer.zero_grad() # clear the gradient
                # wrap them in a torch Variable and move tnsors to the configured device
                images, captions = Variable(data).to(device), Variable(captions).to(device)                                  
                ################
                # Forward Pass #
                ################
                outputs = model(images, captions[:-1])
                # Backward Pass
                # Calculate gradients
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[2]), 
                    captions.reshape(-1)
                    )
                writer.add_scalar('Training loss', loss.item(), global_step=step)
                step += 1
                #################
                # Backward Pass #
                #################
                loss.backward()
                self.optimizer.step() # Update Weights
                epoch_loss += loss.item() # Calculate total Loss
                
                t_prog_bar.set_description('Epoch {}/{}, Loss: {:.4f}, lr={:.7f}'.format(
                    epoch+1, 
                    n_epochs,
                    loss.item(),
                    lr
                )
                                          )
                
#                 torch.cuda.empty_cache()
                del images, captions, loss
                
            total_loss = epoch_loss/len(train_loader.dataset)
            train_error = 1.0 - correct/len(train_loader.dataset)  # 1 - acc
            
            self.metrics.append(
                Metric(
                    loss=epoch_loss, 
                    train_error=train_error
                )
            )
            
            # Decrease the lr
            scheduler.step()
