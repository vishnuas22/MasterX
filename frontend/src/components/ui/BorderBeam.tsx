/**
 * BorderBeam Component - Animated Rotating Border Effect
 * 
 * Creates a rotating gradient beam effect on element borders.
 * Used for highlighting important UI elements like signup forms.
 * 
 * Based on Magic UI Border Beam component
 * @see https://magicui.design/docs/components/border-beam
 * 
 * @module components/ui/BorderBeam
 */

import React from 'react';

// ============================================================================
// TYPES
// ============================================================================

interface BorderBeamProps {
  /**
   * Size of the beam in pixels
   * @default 50
   */
  size?: number;
  
  /**
   * Animation duration in seconds
   * @default 15
   */
  duration?: number;
  
  /**
   * Animation delay in seconds
   * @default 0
   */
  delay?: number;
  
  /**
   * Border width in pixels
   * @default 7.5
   */
  borderWidth?: number;
  
  /**
   * Starting color of gradient (hex)
   * @default "#ffaa40"
   */
  colorFrom?: string;
  
  /**
   * Ending color of gradient (hex)
   * @default "#9c40ff"
   */
  colorTo?: string;
  
  /**
   * Anchor point in degrees (0-360)
   * @default 360
   */
  anchor?: number;
}

// ============================================================================
// COMPONENT
// ============================================================================

/**
 * BorderBeam - Animated border effect
 * 
 * Creates a smooth rotating gradient beam on the border of its parent element.
 * Parent element must have position: relative and border-radius.
 * 
 * @example
 * ```tsx
 * <div className="relative rounded-lg border">
 *   <h2>Card Content</h2>
 *   <BorderBeam size={70} duration={8} borderWidth={3.5} />
 * </div>
 * ```
 */
export const BorderBeam: React.FC<BorderBeamProps> = ({
  size = 60,
  duration = 15,
  delay = 0,
  borderWidth = 7.5,
  colorFrom = "#ffaa40",
  colorTo = "#9c40ff",
  anchor = 360,
}) => {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
        borderRadius: "inherit",
      }}
    >
      {/* The actual rotating beam */}
      <div
        style={{
          position: "absolute",
          inset: `-${borderWidth}px`,
          borderRadius: "inherit",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 1,
            borderRadius: "inherit",
            padding: `${borderWidth}px`,
            WebkitMask: "linear-gradient(white 0 0) content-box, linear-gradient(white 0 0)",
            WebkitMaskComposite: "xor",
            maskComposite: "exclude",
          }}
        >
          {/* Rotating wrapper */}
          <div
            style={{
              width: "100%",
              height: "100%",
              position: "relative",
              animation: `border-beam-rotate ${duration}s linear infinite`,
              animationDelay: `${delay}s`,
            }}
          >
            {/* The beam light */}
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: `${size * 3}px`,
                height: "200%",
                background: `linear-gradient(to bottom, transparent 0%, ${colorFrom} 20%, ${colorTo} 50%, ${colorFrom} 80%, transparent 100%)`,
                transform: `translate(-50%, -50%) rotate(${anchor}deg)`,
                opacity: 0.6,
              }}
            />
          </div>
        </div>
      </div>
      
      {/* Keyframes animation */}
      <style>{`
        @keyframes border-beam-rotate {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

// ============================================================================
// EXPORTS
// ============================================================================

export default BorderBeam;
