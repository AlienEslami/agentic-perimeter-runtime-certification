"""Reproduce the accepted study into a new directory, preserving archives."""
from pathlib import Path
import argparse
import study
import evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit('Choose a new output directory; existing results are preserved.')
    out.mkdir(parents=True)
    study.OUT = out
    evaluate.OUT = out
    for weight, name in [(1.0, 'monotone_backup_class_d25_w300_proof.json.gz'),
                         (0.45, 'monotone_repair045_class_d25_w300_proof.json.gz')]:
        net = study.candidate(weight)
        net.refined = True
        proof = study.verify_new(net, 'class', 25, 300)
        study.save(name, proof)
        if not proof['accepted']:
            raise RuntimeError('Reproduction did not certify ' + name)
        print('Certificate accepted:', name, flush=True)
    evaluate.run()
    print('Validated results:', out, flush=True)

if __name__ == '__main__':
    main()
