"""
generate_dataset.py
--------------------
Creates a synthetic but realistic dataset of phishing and legitimate
emails and saves it to emails_dataset.csv.

You can replace this with a real dataset (e.g. Kaggle's "Phishing Email
Dataset") as long as it has two columns: 'text' and 'label'
(label = 1 for phishing, 0 for safe/legit).
"""

import random
import pandas as pd

random.seed(42)

# ----------------------------------------------------------------------
# Building blocks used to programmatically generate varied emails
# ----------------------------------------------------------------------

phishing_subjects = [
    "Urgent: Verify Your Account Now",
    "Your Account Has Been Suspended",
    "Action Required: Unusual Login Detected",
    "Final Notice: Payment Failed",
    "Congratulations! You Won a Prize",
    "Security Alert: Update Your Password Immediately",
    "Your PayPal Account Will Be Closed",
    "Bank Alert: Confirm Your Identity",
    "Tax Refund Pending - Claim Now",
    "Your Package Could Not Be Delivered",
]

phishing_bodies = [
    "Dear customer, we noticed suspicious activity on your account. "
    "Click here {url} to verify your identity within 24 hours or your "
    "account will be permanently suspended.",
    "Congratulations! You have been selected to win a free gift card. "
    "Claim your prize now by clicking {url} and entering your bank details.",
    "Your account has been limited. To restore full access please login "
    "immediately at {url} and confirm your password and credit card number.",
    "We detected an unauthorized login attempt from a new device. "
    "Verify it was you by clicking {url} within 12 hours, otherwise your "
    "account will be locked.",
    "Your payment for invoice #4471 has failed. Update your billing "
    "information urgently at {url} to avoid service interruption.",
    "Final reminder: your tax refund of $850 is pending. Submit your "
    "social security number and bank account at {url} to receive it.",
    "Your parcel delivery failed due to incomplete address. Click {url} "
    "and pay a small redelivery fee using your card to reschedule.",
    "Act now! Your subscription will auto-renew at a higher price unless "
    "you cancel immediately at {url} by confirming your login credentials.",
]

phishing_urls = [
    "http://secure-paypa1-verify.com/login",
    "http://bank-account-update.ru/verify",
    "http://amaz0n-security.com/confirm",
    "http://192.168.45.12/verify-account",
    "http://account-update-now.xyz/login",
    "http://win-prize-today.tk/claim",
    "http://apple-id-locked.support/verify",
    "http://refund-irs-gov.online/claim",
]

legit_subjects = [
    "Meeting Reminder for Tomorrow",
    "Your Monthly Newsletter",
    "Project Update - Q3 Report",
    "Invoice for Your Recent Purchase",
    "Welcome to Our Service!",
    "Weekly Team Sync Notes",
    "Your Order Has Shipped",
    "Reminder: Submit Your Timesheet",
    "Conference Schedule Attached",
    "Thank You for Your Feedback",
]

legit_bodies = [
    "Hi team, just a quick reminder that our meeting is scheduled for "
    "tomorrow at 10 AM. Please review the attached agenda before joining.",
    "Hello, thank you for subscribing to our newsletter. Here is a "
    "summary of this month's top articles and product updates.",
    "Hi all, attached is the Q3 project report. Please review the metrics "
    "section and share your feedback by Friday.",
    "Dear customer, thank you for your purchase. Your invoice is "
    "attached for your records. Let us know if you have any questions.",
    "Welcome aboard! We're excited to have you. Here is a quick guide to "
    "get started with your new account and explore our features.",
    "Hi team, here are this week's sync notes including action items "
    "and owners. Let me know if I missed anything important.",
    "Good news, your order #88231 has shipped and should arrive within "
    "3-5 business days. You can track it using the link in your account.",
    "Hi, this is a friendly reminder to submit your timesheet for this "
    "week before the Friday deadline. Thanks for staying on top of it.",
    "Hello everyone, please find attached the schedule for next week's "
    "conference sessions. Let us know if you have any dietary requirements.",
    "Thank you for sharing your feedback with us. We truly value your "
    "input and will use it to improve our product going forward.",
]

legit_urls = [
    "https://www.company.com/dashboard",
    "https://docs.google.com/document/d/abc123",
    "https://github.com/yourorg/project",
    "https://www.linkedin.com/in/yourprofile",
    "https://mail.yourcompany.com/inbox",
    "",  # many legit emails have no link at all
    "",
    "",
]


def make_email(subject, body_template, url_pool):
    url = random.choice(url_pool)
    body = body_template.format(url=url) if "{url}" in body_template else body_template
    if url and "{url}" not in body_template:
        body += f" More info: {url}"
    return f"Subject: {subject}\n{body}"


def build_dataset(n_per_class=300):
    rows = []
    for _ in range(n_per_class):
        subj = random.choice(phishing_subjects)
        body = random.choice(phishing_bodies)
        text = make_email(subj, body, phishing_urls)
        rows.append({"text": text, "label": 1})  # 1 = phishing

    for _ in range(n_per_class):
        subj = random.choice(legit_subjects)
        body = random.choice(legit_bodies)
        text = make_email(subj, body, legit_urls)
        rows.append({"text": text, "label": 0})  # 0 = safe

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = build_dataset(n_per_class=300)
    df.to_csv("emails_dataset.csv", index=False)
    print(f"✅ Dataset created: emails_dataset.csv  ({len(df)} rows)")
    print(df['label'].value_counts())
