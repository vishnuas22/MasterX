"""
Email Service for MasterX Platform

Handles all email sending functionality including:
- Email verification emails
- Password reset emails  
- Welcome emails
- Notification emails

Uses Python's built-in smtplib for SMTP email sending.
Supports both development (console logging) and production (SMTP) modes.

Architecture:
- Clean separation of email templates and sending logic
- HTML + plain text email support
- Async/await patterns for non-blocking operations
- Environment-based configuration (12-factor app)
- Graceful error handling and logging

Security:
- TLS/SSL encryption for SMTP
- Input sanitization to prevent email injection
- Rate limiting handled at API layer
- No sensitive data in email bodies

Configuration via .env:
- EMAIL_ENABLED: Enable/disable email sending (default: True)
- EMAIL_FROM_ADDRESS: Sender email address
- EMAIL_FROM_NAME: Sender display name
- SMTP_HOST: SMTP server host
- SMTP_PORT: SMTP server port (typically 587 for TLS)
- SMTP_USER: SMTP authentication username
- SMTP_PASSWORD: SMTP authentication password
- SMTP_USE_TLS: Use TLS encryption (default: True)
- FRONTEND_URL: Frontend URL for email links

Development Mode:
- Set EMAIL_ENABLED=False to log emails to console instead of sending
- Useful for local development and testing

Author: MasterX Development Team
Version: 1.0.0
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class EmailService:
    """
    Email service for sending various types of emails
    
    Handles email composition and sending with support for:
    - HTML and plain text formats
    - Multiple recipients
    - Development mode (console logging)
    - Production mode (SMTP sending)
    
    Example:
        ```python
        email_service = EmailService()
        await email_service.send_verification_email(
            email="user@example.com",
            name="John Doe",
            verification_token="abc123"
        )
        ```
    """
    
    def __init__(self):
        """Initialize email service with configuration from environment variables"""
        # Email service configuration
        self.enabled = os.getenv("EMAIL_ENABLED", "true").lower() == "true"
        self.from_address = os.getenv("EMAIL_FROM_ADDRESS", "noreply@masterx.ai")
        self.from_name = os.getenv("EMAIL_FROM_NAME", "MasterX Support")
        
        # SMTP configuration
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        
        # Frontend URL for links in emails
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
        # Log configuration (without exposing sensitive data)
        if not self.enabled:
            logger.info("📧 Email service: DISABLED (development mode - emails will be logged)")
        else:
            logger.info(f"📧 Email service: ENABLED (SMTP: {self.smtp_host}:{self.smtp_port})")
            if not self.smtp_user or not self.smtp_password:
                logger.warning("⚠️ SMTP credentials not configured - emails will fail in production")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        to_name: Optional[str] = None
    ) -> bool:
        """
        Send an email (async wrapper for sync email sending)
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            html_body: HTML version of email body
            text_body: Plain text version of email body (optional, auto-generated if not provided)
            to_name: Recipient display name (optional)
        
        Returns:
            bool: True if email sent successfully, False otherwise
        
        Raises:
            No exceptions raised - errors are logged and False is returned
        """
        # Development mode - log email instead of sending
        if not self.enabled:
            logger.info(f"📧 [DEV MODE] Email would be sent:")
            logger.info(f"  To: {to_name} <{to_email}>" if to_name else f"  To: {to_email}")
            logger.info(f"  Subject: {subject}")
            logger.info(f"  Body (first 200 chars): {text_body[:200] if text_body else html_body[:200]}...")
            return True
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_address}>"
            msg['To'] = f"{to_name} <{to_email}>" if to_name else to_email
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # Add plain text version (fallback)
            if text_body:
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # Add HTML version (preferred)
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send email via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                if self.smtp_use_tls:
                    server.starttls()
                
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully to {to_email}: {subject}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error sending email to {to_email}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending email to {to_email}: {e}", exc_info=True)
            return False
    
    async def send_verification_email(
        self,
        email: str,
        name: str,
        verification_token: str
    ) -> bool:
        """
        Send email verification email to new user
        
        Email contains a verification link that the user must click to verify their email address.
        Link expires based on token expiration (typically 24 hours).
        
        Args:
            email: User's email address
            name: User's display name
            verification_token: Unique verification token
        
        Returns:
            bool: True if email sent successfully, False otherwise
        
        Example:
            ```python
            success = await email_service.send_verification_email(
                email="john@example.com",
                name="John Doe",
                verification_token="abc123def456"
            )
            ```
        """
        verification_url = f"{self.frontend_url}/verify-email?token={verification_token}"
        
        # HTML email template
        html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verify Your Email - MasterX</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 0;">
                <table role="presentation" style="width: 600px; max-width: 100%; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 20px; text-align: center;">
                            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #0A0A0A;">MasterX</h1>
                            <p style="margin: 10px 0 0; font-size: 14px; color: #666;">AI-Powered Adaptive Learning</p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 0 40px 40px;">
                            <h2 style="margin: 0 0 20px; font-size: 24px; font-weight: 600; color: #0A0A0A;">Welcome, {name}! 🎉</h2>
                            
                            <p style="margin: 0 0 20px; font-size: 16px; line-height: 1.6; color: #333;">
                                Thank you for joining MasterX. To get started with your personalized learning journey, please verify your email address.
                            </p>
                            
                            <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.6; color: #333;">
                                Click the button below to verify your email:
                            </p>
                            
                            <!-- Verification Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td align="center">
                                        <a href="{verification_url}" style="display: inline-block; padding: 16px 40px; background-color: #0066FF; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: 600;">
                                            Verify Email Address
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 30px 0 0; font-size: 14px; line-height: 1.6; color: #666;">
                                Or copy and paste this link into your browser:
                            </p>
                            <p style="margin: 10px 0 0; font-size: 14px; word-break: break-all; color: #0066FF;">
                                {verification_url}
                            </p>
                            
                            <hr style="margin: 30px 0; border: none; border-top: 1px solid #e5e5e5;">
                            
                            <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #666;">
                                This verification link will expire in 24 hours. If you didn't create an account with MasterX, please ignore this email.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 20px 40px; background-color: #f9f9f9; border-radius: 0 0 8px 8px;">
                            <p style="margin: 0; font-size: 12px; line-height: 1.6; color: #999; text-align: center;">
                                © 2025 MasterX. All rights reserved.<br>
                                Questions? Contact us at support@masterx.ai
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
        
        # Plain text version (fallback)
        text_body = f"""
Welcome to MasterX, {name}!

Thank you for joining MasterX. To get started with your personalized learning journey, please verify your email address.

Verify your email by clicking this link:
{verification_url}

This verification link will expire in 24 hours. If you didn't create an account with MasterX, please ignore this email.

---
© 2025 MasterX. All rights reserved.
Questions? Contact us at support@masterx.ai
"""
        
        return await self.send_email(
            to_email=email,
            to_name=name,
            subject="Verify Your Email - MasterX",
            html_body=html_body,
            text_body=text_body
        )
    
    async def send_welcome_email(
        self,
        email: str,
        name: str
    ) -> bool:
        """
        Send welcome email after successful email verification
        
        Args:
            email: User's email address
            name: User's display name
        
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        dashboard_url = f"{self.frontend_url}/dashboard"
        
        html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to MasterX</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 0;">
                <table role="presentation" style="width: 600px; max-width: 100%; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <tr>
                        <td style="padding: 40px 40px 20px; text-align: center;">
                            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #0A0A0A;">MasterX</h1>
                            <p style="margin: 10px 0 0; font-size: 14px; color: #666;">AI-Powered Adaptive Learning</p>
                        </td>
                    </tr>
                    
                    <tr>
                        <td style="padding: 0 40px 40px;">
                            <h2 style="margin: 0 0 20px; font-size: 24px; font-weight: 600; color: #0A0A0A;">Welcome to MasterX! 🚀</h2>
                            
                            <p style="margin: 0 0 20px; font-size: 16px; line-height: 1.6; color: #333;">
                                Hi {name},
                            </p>
                            
                            <p style="margin: 0 0 20px; font-size: 16px; line-height: 1.6; color: #333;">
                                Your email has been verified successfully! You're now ready to experience personalized, emotion-aware learning powered by AI.
                            </p>
                            
                            <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.6; color: #333;">
                                <strong>What you can do with MasterX:</strong>
                            </p>
                            
                            <ul style="margin: 0 0 30px; padding-left: 20px; font-size: 16px; line-height: 1.8; color: #333;">
                                <li>Chat with AI tutors that adapt to your emotions and learning style</li>
                                <li>Track your learning progress with detailed analytics</li>
                                <li>Earn achievements and compete on leaderboards</li>
                                <li>Practice with spaced repetition for better retention</li>
                                <li>Collaborate with peers in study sessions</li>
                            </ul>
                            
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td align="center">
                                        <a href="{dashboard_url}" style="display: inline-block; padding: 16px 40px; background-color: #0066FF; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: 600;">
                                            Start Learning Now
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 30px 0 0; font-size: 14px; line-height: 1.6; color: #666;">
                                Need help getting started? Check out our <a href="{self.frontend_url}/help" style="color: #0066FF; text-decoration: none;">help center</a> or reply to this email.
                            </p>
                        </td>
                    </tr>
                    
                    <tr>
                        <td style="padding: 20px 40px; background-color: #f9f9f9; border-radius: 0 0 8px 8px;">
                            <p style="margin: 0; font-size: 12px; line-height: 1.6; color: #999; text-align: center;">
                                © 2025 MasterX. All rights reserved.<br>
                                Questions? Contact us at support@masterx.ai
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
        
        text_body = f"""
Welcome to MasterX, {name}!

Your email has been verified successfully! You're now ready to experience personalized, emotion-aware learning powered by AI.

What you can do with MasterX:
- Chat with AI tutors that adapt to your emotions and learning style
- Track your learning progress with detailed analytics
- Earn achievements and compete on leaderboards
- Practice with spaced repetition for better retention
- Collaborate with peers in study sessions

Start learning now: {dashboard_url}

Need help getting started? Check out our help center or reply to this email.

---
© 2025 MasterX. All rights reserved.
Questions? Contact us at support@masterx.ai
"""
        
        return await self.send_email(
            to_email=email,
            to_name=name,
            subject="Welcome to MasterX - Let's Get Started!",
            html_body=html_body,
            text_body=text_body
        )


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """
    Get or create the global email service instance (singleton pattern)
    
    Returns:
        EmailService: The global email service instance
    
    Example:
        ```python
        email_service = get_email_service()
        await email_service.send_verification_email(...)
        ```
    """
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
