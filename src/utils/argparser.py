import argparse, textwrap

def parse_input():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--method", required=True,
                        choices=["centime", "classical", "cox", "coxmb", "deephit"],
                        help="Select which trainer to run")
                        
    parser.add_argument("--head", required=True,
                        choices=["standard", "interp"],
                        help="Head type to attach to the backbone")
                        
    parser.add_argument("--use-lungmask", action=argparse.BooleanOptionalAction,
                        help="Whether to use lungmask or not. Ignored if --data dummy is set.")
                        
    parser.add_argument("--load-ckpt", action=argparse.BooleanOptionalAction,
                        help="Whether to load the latest checkpoint if available")
    
    parser.add_argument("--data", default="osic",
                        choices=["osic", "dummy1", "dummy2", "dummy3", "dummy4", "dummy5", "dummy6"],
                        help=textwrap.dedent("""\
                            Dataset to use. Complexity levels of the dummy datasets:

                            1: single bright centered spheres, survival time scales with sphere size. Random noise otherwise
                               Expected: model highlights sphere
                               Demonstrates: model can learn simple features by survival times

                            2: as 1, but with random number (1-3) of bright spheres at random locations and of random sizes
                               Expected: model highlights multiple spheres
                               Demonstrates: model can learn feature presence, but not by their fixed location

                            3: many small bright bubbles only (near edge of image), simulating cysts in IPF
                               Expected: model highlights bubbles
                               Demonstrates: model can learn to focus on small relevant features

                            4: as 1, but sphere is of average brightness vs rest of image
                               Expected: model highlights sphere
                               Demonstrates: model can learn features by shape & placement, not only by excessive brightness. Real fibrosis is not excessively bright in CT scans.

                            5: as 2 & 3, but spheres for short survival time samples & random bubbles in all samples
                               Expected: model highlights spheres, ignores bubbles
                               Demonstrates: model can learn to focus on relevant features & ignore tempting confounders of similar appearance

                            6: as 5, but sphere of average brightness always present in corner.
                               Expected: model highlights spheres, ignores bubbles & corner sphere
                               Demonstrates: model can learn to ignore tempting confounders of similar appearance & fixed location. Model should be able to highlight IPF-causing features, ignore similar-looking non-IPF-causing features & ignore omnipresent structures (trachea, heart) if irrelevant. 
                            """))
                        
    return parser.parse_args()