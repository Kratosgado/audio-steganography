{
  description = "Python + NumPy shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShell.${system} = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          python312Packages.numpy
          python312Packages.librosa
          python312Packages.scipy
          python312Packages.fastapi
          python312Packages.uvloop
          python312Packages.stable-baselines3
          python312Packages.shimmy
          python312Packages.soundfile
          python312Packages.matplotlib
          python312Packages.torch
          python312Packages.scikit-learn
        ];
      };
    };
}
