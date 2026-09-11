"""Extract directly observed ChArUco corners; no completion or TPS smoothing."""
from pathlib import Path
import argparse, hashlib, json, re, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


def detect(path):
    cv2.setNumThreads(1)
    board = cv2.aruco.CharucoBoard((16, 12), .3, .15,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    board.setLegacyPattern(True)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 75, 4
    p.minMarkerPerimeterRate, p.maxMarkerPerimeterRate = .005, .20
    p.polygonalApproxAccuracyRate, p.minCornerDistanceRate = .03, .02
    p.minDistanceToBorder, p.errorCorrectionRate = 1, .6
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    cp = cv2.aruco.CharucoParameters(); cp.checkMarkers = False
    detector = cv2.aruco.CharucoDetector(board, cp, p)
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError(path)
    corners, ids, _, _ = detector.detectBoard(im)
    result = np.full((165, 2), np.nan)
    if ids is not None:
        cv2.cornerSubPix(im, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4))
        result[ids.ravel()] = corners.reshape(-1, 2)
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def depth(path):
    return float(re.findall(r'[-+]?\d*\.?\d+', path.stem)[-1])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--pycaso', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('paper/strain/results'))
    p.add_argument('--workers', type=int, default=4); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    root = a.pycaso / 'Exemple/Images_example'
    for series in ('calibration', 'calibration2'):
        maps = [{depth(f): f for f in (root / (side + '_' + series)).iterdir() if f.suffix.lower() in ('.tif','.tiff','.png')} for side in ('left', 'right')]
        z = sorted(set(maps[0]) & set(maps[1]))
        assert len(z) == 101, (series, len(z))
        files = [m[v] for v in z for m in maps]
        t = time.perf_counter()
        with ThreadPoolExecutor(a.workers) as pool: obs = list(pool.map(detect, files))
        pixels = np.stack([x[0] for x in obs]).reshape(len(z), 2, 165, 2)
        board = cv2.aruco.CharucoBoard((16,12),.3,.15,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
        board.setLegacyPattern(True)
        np.savez_compressed(a.out / f'observations_{series}.npz', pixels=pixels, z=z, xy=board.getChessboardCorners()[:,:2])
        common = np.isfinite(pixels).all(axis=(1,3)).sum(axis=1)
        manifest = dict(series=series, frames=len(z), common_min=int(common.min()), common_median=float(np.median(common)),
            common_max=int(common.max()), opencv=cv2.__version__, elapsed_s=time.perf_counter()-t,
            files=[dict(path=str(f.relative_to(a.pycaso)), sha256=o[1]) for f,o in zip(files,obs)])
        (a.out / f'detection_{series}.json').write_text(json.dumps(manifest,indent=2))
        print(series, {k:v for k,v in manifest.items() if k!='files'},flush=True)

if __name__ == '__main__': main()
