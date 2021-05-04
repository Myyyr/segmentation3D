import alltrain
import  argparse

import importlib




def main(config, tensorboard):
	
	print(config)
	cfg = importlib.import_module(config)
	excfg = cfg.ExpConfig()

	train = alltrain.AllTrain(excfg, tensorboard)
	train.train()

	return 0

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)
	parser.add_argument('-t', "--tensorboard", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate tensorboard view.")

	main(parser.parse_args().config, parser.parse_args().tensorboard)


