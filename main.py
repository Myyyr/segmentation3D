import alltrain
import  argparse

import importlib




def main(config):
	if config == 'example':
		import expconfigs.example as cfg


		excfg = cfg.ExpConfig()

		train = alltrain.BTrain(excfg)

		train.train()

		return 0

	if config == 'unet':
		import expconfigs.unet as cfg


		excfg = cfg.ExpConfig()

		train = alltrain.BTrain(excfg)

		train.train()

		return 0

	if config == 'revunet':
		import expconfigs.revunet as cfg


		excfg = cfg.ExpConfig()

		train = alltrain.BTrain(excfg)

		train.train()

		return 0


	if config == "multi_atlas_revunet":
		import expconfigs.multi_atlas_revunet as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "multi_atlas_unet":
		import expconfigs.multi_atlas_unet as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0


	if config == "memory_multi_atlas":
		import expconfigs.memory_multi_atlas_revunet as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MemMATrain(excfg)
		train.train()

		return 0


	if config == "ma_unet_res01":
		import expconfigs.multi_atlas_unet_res01 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "ma_unet_res01_v2":
		import expconfigs.multi_atlas_unet_res01_v2 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "ma_unet_res01_v3":
		import expconfigs.multi_atlas_unet_res01_v3 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "multi_atlas_revunet_1_v1":
		import expconfigs.multi_atlas_revunet_1_v1 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "multi_atlas_revunet_5_v1":
		import expconfigs.multi_atlas_revunet_5_v1 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "multi_atlas_revunet_01_v1":
		import expconfigs.multi_atlas_revunet_01_v1 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0

	if config == "multi_atlas_revunet_01_v2":
		import expconfigs.multi_atlas_revunet_01_v2 as cfg
		excfg = cfg.ExpConfig()

		train = alltrain.MATrain(excfg)
		train.train()

		return 0
	
	cfg = importlib.import_module("expconfigs."+config+".py")
	excfg = cfg.ExpConfig()

	train = alltrain.MATrain(excfg)
	train.train()

	return 0

if __name__ == "__main__" :
	parser = argparse.ArgumentParser("Configs for training")
	parser.add_argument('-c', '--config', help='Config to use', required=True)

	main(parser.parse_args().config)


