from torch.optim import lr_scheduler





def get_scheduler(optimize, opt):
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

    return scheduler


