from torch.optim import lr_scheduler





def get_scheduler(optimizer, opt, lr, decay=None):
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
            if epoch < 300:
                lr_l = lr[0]
            elif 300 <= epoch < 650:
                lr_l = lr[1]
            else:
                lr_l = lr[2]
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

    if opt == "multistep":
    
        scheduler = lr_scheduler.MultiStepLR(optimizer, [250, 500, 750], 0.2) 

    if opt == "multistep1000":
    
        scheduler = lr_scheduler.MultiStepLR(optimizer, [300, 600, 900, 1200], 0.2) 
    

    if opt == "constant":
        def lambda_rule(epoch):
            #print(epoch)
            return lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

    if opt == "po":
        def lambda_rule(epoch):
            return 1/(1 + decay*epoch)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 
    return scheduler


