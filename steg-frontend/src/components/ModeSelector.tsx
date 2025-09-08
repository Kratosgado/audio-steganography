import React from "react";

interface ModeSelectorProps {
	selected: string;
	onModeChange: (mode: string) => void;
}

export function ModeSelector({ selected, onModeChange }: ModeSelectorProps) {
	return (
		<div style={{ display: "flex", justifyContent: "center" }}>
			<div className="radio-inputs">
				{["Encode", "Decode"].map((option) => (
					<label key={option} className="radio">
						<input
							type="radio"
							name="radio"
							value={option}
							checked={selected === option}
							onChange={() => onModeChange(option)}
						/>
						<span className="name">{option}</span>
					</label>
				))}
			</div>

			<style jsx>{`
        .radio-inputs {
          position: relative;
          display: flex;
          flex-wrap: wrap;
          border-radius: 0.5rem;
          background-color: #eee;
          box-sizing: border-box;
          box-shadow: 0 0 0px 1px rgba(0, 0, 0, 0.06);
          padding: 0.25rem;
          width: 300px;
          font-size: 14px;
          margin: 1rem 0;
        }

        .radio-inputs .radio {
          flex: 1 1 auto;
          text-align: center;
        }

        .radio-inputs .radio input {
          display: none;
        }

        .radio-inputs .radio .name {
          display: flex;
          cursor: pointer;
          align-items: center;
          justify-content: center;
          border-radius: 0.5rem;
          border: none;
          padding: 0.5rem 0;
          color: rgba(51, 65, 85, 1);
          transition: all 0.15s ease-in-out;
        }

        .radio-inputs .radio input:checked + .name {
          background-color: #fff;
          font-weight: 600;
        }
      `}</style>
		</div>
	);
}
