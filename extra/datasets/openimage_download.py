import os
import json
import argparse
import fiftyone as fo
import fiftyone.zoo as foz

MLPERF_CLASSES=['Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
  'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench', 'Bicycle',
  'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book', 'Bookcase', 'Boot',
  'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread',
  'Broccoli', 'Bronze sculpture', 'Bull', 'Bus', 'Bust', 'Butterfly', 'Cabinetry',
  'Cake', 'Camel', 'Camera', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart',
  'Castle', 'Cat', 'Cattle', 'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
  'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup', 'Coffee table', 'Coin',
  'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Convenience store',
  'Cookie', 'Countertop', 'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard',
  'Curtain', 'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door', 'Dragonfly',
  'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant',
  'Falcon', 'Fedora', 'Flag', 'Flowerpot', 'Football', 'Football helmet', 'Fork',
  'Fountain', 'French fries', 'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses',
  'Goat', 'Goggles', 'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
  'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter', 'High heels',
  'Hiking equipment', 'Horse', 'House', 'Houseplant', 'Human arm', 'Human beard',
  'Human body', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
  'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream',
  'Jacket', 'Jeans', 'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite',
  'Lamp', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb', 'Lighthouse',
  'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple', 'Microphone', 'Mirror',
  'Mixing bowl', 'Mobile phone', 'Monkey', 'Motorcycle', 'Muffin', 'Mug', 'Mule',
  'Mushroom', 'Musical keyboard', 'Necklace', 'Nightstand', 'Office building',
  'Orange', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen',
  'Penguin', 'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
  'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit', 'Rifle',
  'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Sea lion',
  'Sea turtle', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard',
  'Ski', 'Skull', 'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
  'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry', 'Street light',
  'Studio couch', 'Suit', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan',
  'Swimming pool', 'Swimwear', 'Tank', 'Tap', 'Taxi', 'Tea', 'Teddy bear', 'Television',
  'Tent', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
  'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella', 'Van', 'Vase',
  'Vehicle registration plate', 'Violin', 'Wall clock', 'Waste container', 'Watch',
  'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Window', 'Wine', 'Wine glass', 'Woman',
  'Zebra', 'Zucchini']

parser = argparse.ArgumentParser(description='Download OpenImages using FiftyOne', add_help=True)
parser.add_argument('--dataset-dir', default='extra/datasets/open-images-v6TEST', help='dataset download location')
parser.add_argument('--splits', default=['train', 'validation'], choices=['train', 'validation', 'test'],
                    nargs='+', type=str,
                    help='Splits to download, possible values are train, validation and test')
parser.add_argument('--classes', default=MLPERF_CLASSES, nargs='+', type=str,
                    help='Classes to download. default to all classes')
parser.add_argument('--output-labels', default='openimages-mlperf.json', type=str,
                    help='Classes to download. default to all classes')
args = parser.parse_args()

dir = args.dataset_dir

print("Downloading open-images dataset ...")
dataset = foz.load_zoo_dataset(
	name="open-images-v6",
	classes=args.classes,
	splits=args.splits,
	label_types=["detections"],
	dataset_name="open-images",
	dataset_dir=dir,
	# max_samples=1000,
)

print("Converting dataset to coco format ...")
for split in args.splits:
	output_fname = os.path.join(dir, split, "labels", args.output_labels)
	split_view = dataset.match_tags(split)
	split_view.export(
		labels_path=output_fname,
		dataset_type=fo.types.COCODetectionDataset,
		classes=args.classes)

	# Add iscrowd label to openimages annotations
	with open(output_fname) as fp:
		labels = json.load(fp)
	for annotation in labels['annotations']:
		annotation['iscrowd'] = int(annotation['IsGroupOf'])
	with open(output_fname, "w") as fp:
		json.dump(labels, fp)