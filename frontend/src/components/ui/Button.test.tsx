/**
 * Button.test.tsx - Unit Tests for Button Component
 * 
 * Purpose: Test button component variants and interactions
 * 
 * Coverage:
 * - Rendering variants
 * - Click handlers
 * - Disabled state
 * - Loading state
 * - Accessibility
 * 
 * Following AGENTS_FRONTEND.md:
 * - Test coverage > 80%
 * - Accessibility testing
 * - User interaction testing
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/testUtils';
import { Button } from '@/components/ui/Button';
import userEvent from '@testing-library/user-event';

describe('Button Component', () => {
  // ============================================================================
  // RENDERING
  // ============================================================================

  describe('Rendering', () => {
    it('should render button with text', () => {
      render(<Button>Click me</Button>);
      
      expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
    });

    it('should render primary variant', () => {
      render(<Button variant="primary">Primary</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-blue-500');
    });

    it('should render secondary variant', () => {
      render(<Button variant="secondary">Secondary</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-gray-700');
    });

    it('should render outline variant', () => {
      render(<Button variant="outline">Outline</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('border');
    });

    it('should render different sizes', () => {
      const { rerender } = render(<Button size="sm">Small</Button>);
      expect(screen.getByRole('button')).toHaveClass('px-3', 'py-1.5');

      rerender(<Button size="md">Medium</Button>);
      expect(screen.getByRole('button')).toHaveClass('px-4', 'py-2');

      rerender(<Button size="lg">Large</Button>);
      expect(screen.getByRole('button')).toHaveClass('px-6', 'py-3');
    });
  });

  // ============================================================================
  // INTERACTIONS
  // ============================================================================

  describe('Interactions', () => {
    it('should call onClick handler when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<Button onClick={handleClick}>Click me</Button>);
      
      const button = screen.getByRole('button');
      await user.click(button);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('should not call onClick when disabled', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<Button onClick={handleClick} disabled>Click me</Button>);
      
      const button = screen.getByRole('button');
      await user.click(button);

      expect(handleClick).not.toHaveBeenCalled();
    });

    it('should not call onClick when loading', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<Button onClick={handleClick} loading>Click me</Button>);
      
      const button = screen.getByRole('button');
      await user.click(button);

      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  // ============================================================================
  // DISABLED STATE
  // ============================================================================

  describe('Disabled State', () => {
    it('should have disabled attribute when disabled', () => {
      render(<Button disabled>Disabled</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('should have opacity class when disabled', () => {
      render(<Button disabled>Disabled</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('opacity-50');
    });
  });

  // ============================================================================
  // LOADING STATE
  // ============================================================================

  describe('Loading State', () => {
    it('should show loading spinner when loading', () => {
      render(<Button loading>Loading</Button>);
      
      // Look for loading spinner
      const button = screen.getByRole('button');
      expect(button.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('should be disabled when loading', () => {
      render(<Button loading>Loading</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });
  });

  // ============================================================================
  // ACCESSIBILITY
  // ============================================================================

  describe('Accessibility', () => {
    it('should have correct role', () => {
      render(<Button>Accessible</Button>);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('should support aria-label', () => {
      render(<Button aria-label="Close dialog">X</Button>);
      
      expect(screen.getByLabelText('Close dialog')).toBeInTheDocument();
    });

    it('should be keyboard accessible', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<Button onClick={handleClick}>Press Enter</Button>);
      
      const button = screen.getByRole('button');
      button.focus();
      expect(button).toHaveFocus();

      await user.keyboard('{Enter}');
      expect(handleClick).toHaveBeenCalled();
    });

    it('should have visible focus indicator', () => {
      render(<Button>Focus me</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('focus:ring-2');
    });
  });

  // ============================================================================
  // FULL WIDTH
  // ============================================================================

  describe('Full Width', () => {
    it('should render full width when specified', () => {
      render(<Button fullWidth>Full Width</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('w-full');
    });
  });

  // ============================================================================
  // CUSTOM CLASS
  // ============================================================================

  describe('Custom Class', () => {
    it('should accept custom className', () => {
      render(<Button className="custom-class">Custom</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('custom-class');
    });
  });

  // ============================================================================
  // BUTTON TYPE
  // ============================================================================

  describe('Button Type', () => {
    it('should default to button type', () => {
      render(<Button>Default</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'button');
    });

    it('should support submit type', () => {
      render(<Button type="submit">Submit</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'submit');
    });

    it('should support reset type', () => {
      render(<Button type="reset">Reset</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'reset');
    });
  });
});
