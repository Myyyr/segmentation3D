import data_process
import  argparse





def main(config, s, e):
	if config == 'mat':
		import expconfigs.multi_atlas_unet as cfg


		excfg = cfg.ExpConfig()

		train = data_process.LookMAT(excfg)

		train.train(s, e)

		return 0

	

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)
	parser.add_argument('-s', '--start', help='start image', required=True)
	parser.add_argument('-e', '--end', help='last image', required=True)

	main(parser.parse_args().config, int(parser.parse_args().start), int(parser.parse_args().end))


