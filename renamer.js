/**
 * Created on 1398/11/16 (2020/2/5).
 * @author {@link https://mirismaili.github.io S. Mahdi Mir-Ismaili}
 */
'use strict'

const fs = require('fs')

// process.chdir('../bashgah-more-captcha/wrongs')
process.chdir('./captchas/1')

for (const name of fs.readdirSync('.'))
	fs.rename(name,
			`${name.substr(0, 5)}.jpg`,
			err => {
				if (err) console.error(err)
			})
