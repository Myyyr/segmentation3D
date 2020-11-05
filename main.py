import alltrain.BratsTrain as BratsTrain
import  argparse





def main(config):
	if config == 'example':
		import expconfigs.example as cfg


		excfg = cfg.ExpConfig()

		train = BratsTrain.BTrain(excfg)

		train.train()





if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


