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
          python311
          python311Packages.numpy
          python311Packages.ipykernel
          python311Packages.librosa
          python311Packages.scipy
          python311Packages.soundfile
          python311Packages.matplotlib
          python311Packages.tensorflow
          python311Packages.scikit-learn
        ];
      };
    };
}
