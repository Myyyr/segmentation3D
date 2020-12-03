import data_process
import  argparse





def main(config, s, e, mod):
	if config == 'mat':
		import expconfigs.multi_atlas_unet as cfg


		excfg = cfg.ExpConfig()

		train = data_process.LookMAT(excfg)

		if mod == 'look':
			train.train(s, e)
		elif mod == 'info':
			train.data_info(e)
		return 0

	

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)
	parser.add_argument('-s', '--start', help='start image', required=True)
	parser.add_argument('-e', '--end', help='last image', required=True)
	parser.add_argument('-m', '--mode', help='data look mode', required=True)

	main(parser.parse_args().config, int(parser.parse_args().start), int(parser.parse_args().end), parser.parse_args().mode)


