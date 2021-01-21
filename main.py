import alltrain
import  argparse

import importlib




def main(config):
	
	print("expconfigs."+config)
	cfg = importlib.import_module("expconfigs."+config)
	excfg = cfg.ExpConfig()

	train = alltrain.AllTrain(excfg)
	train.train()

	return 0

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


