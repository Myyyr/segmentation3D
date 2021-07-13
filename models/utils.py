from torch.optim import lr_scheduler





def get_scheduler(optimizer, opt, lr, decay=None, max_epochs=None):
    scheduler = None
    if opt == "lambdarule_1":
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 60:
                lr_l = 0.01
            else:
                lr_l = 0.002
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 


    if opt == "lambdarule_e1000":
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 1100:
                lr_l = 1
            elif 1100 <= epoch < 1200:
                lr_l = 0.5
            elif 1200 <= epoch < 1500:
                lr_l = 0.1
            return 0.01
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

    if opt == "multistep":
    
        scheduler = lr_scheduler.MultiStepLR(optimizer, [250, 500, 750], 0.2) 

    if opt == "multistep1000":
    
        scheduler = lr_scheduler.MultiStepLR(optimizer, [300, 600, 900, 1200], 0.2) 
    

    if opt == "constant":
        def lambda_rule(epoch):
            #print(epoch)
            return 1
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

    if opt == "po":
        def lambda_rule(epoch):
            return 1/(1 + decay*epoch)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 
    
    if opt == 'poly':
        def lambda_rule(epoch):
            return (1 - epoch / max_epochs)**0.9
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

        





    return scheduler    