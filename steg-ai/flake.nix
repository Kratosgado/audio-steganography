
{
  description = "Python + NumPy shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShell.${system} = pkgs.mkShell {
        buildInputs = [
          pkgs.python311
          pkgs.python311Packages.numpy
          pkgs.python311Packages.ipykernel
        ];
      };
    };
}
