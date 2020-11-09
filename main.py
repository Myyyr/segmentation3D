import alltrain
import  argparse





def main(config):
	if config == 'example':
		import expconfigs.example as cfg


		excfg = cfg.ExpConfig()

		train = alltrain.BTrain(excfg)

		train.train()

	if config == 'unet':
		import expconfigs.unet as cfg


		excfg = cfg.ExpConfig()

		train = alltrain.BTrain(excfg)

		train.train()





if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


