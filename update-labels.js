/**
 * Created on 1398/11/14 (2020/2/3).
 * @author {@link https://mirismaili.github.io S. Mahdi Mir-Ismaili}
 */
const fs = require('fs')

fs.writeFileSync('./labels.js',
		'/**X\n' +
		' * Created on 1398/11/14 (2020/2/3).\n' +
		' * @author {@link https://mirismaili.github.io S. Mahdi Mir-Ismaili}\n' +
		' */\n' +
		'\'use strict\'\n' +
		'\n' +
		'export default ' +
		JSON.stringify(
				fs.readdirSync('./captchas/1').map(name => name.substr(0,5)),
				null, '\t')
				.replace(/"/g, "'")
)
