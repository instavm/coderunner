import { chromium  } from 'playwright';
import { expect } from 'playwright/test';

const RemoteChrome= async () => {
const browser = await chromium.connect('ws://play20.local:3000/');
const page = await browser.newPage();

await page.goto('https://www.amazon.com')
await expect(page).toHaveURL(/amazon/)
return await page.title()
};

const op = await RemoteChrome()
console.log(op)
