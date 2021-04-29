import alltrain
import  argparse

import importlib




def main(config):
	
	print(config)
	cfg = importlib.import_module(config)
	excfg = cfg.ExpConfig()

	train = alltrain.AllTrain(excfg)
	train.evaluate(True)

	return 0

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


