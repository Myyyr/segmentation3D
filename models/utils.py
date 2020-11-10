from torch.optim import lr_scheduler





def get_scheduler(optimizer, opt):
    scheduler = None
    if opt == "lambdarule_1":
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 60:
                lr_l = 0.01
            elif 60 <= epoch < 101:
                lr_l = 0.002
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 


    if opt == "lambdarule_e1000":
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 300:
                lr_l = 0.01
            elif 300 <= epoch < 650:
                lr_l = 0.002
            else:
                lr_l = 0.0004
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) 

    return scheduler

    if opt == "multistep":
    
        scheduler = lr_scheduler.MultiStepLR(optimizer, [250, 400, 550], 0.2) 

    return scheduler


