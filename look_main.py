import data_process
import  argparse





def main(config):
	if config == 'mat':
		import expconfigs.multi_atlas_unet as cfg


		excfg = cfg.ExpConfig()

		train = data_process.LookMAT(excfg)

		train.train()

		return 0

	

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


