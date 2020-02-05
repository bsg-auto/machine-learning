/**
 * Created on 1398/11/14 (2020/2/3).
 * @author {@link https://mirismaili.github.io S. Mahdi Mir-Ismaili}
 */
const fs = require('fs')

fs.writeFileSync('./labels.js',
		'/**\n' +
		' * Created on 1398/11/14 (2020/2/3).\n' +
		' * @author {@link https://mirismaili.github.io S. Mahdi Mir-Ismaili}\n' +
		' */\n' +
		'\'use strict\'\n' +
		'\n' +
		'export default ' +
		JSON.stringify(
				shuffle(fs.readdirSync('./captchas/1')).map(name => name.substr(0,11)),
				null, '\t')
				.replace(/"/g, "'") +
		'\n'
)

/**
 * Shuffles array in place. ES6 version
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[a[i], a[j]] = [a[j], a[i]];
	}
	return a;
}
